from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minifrontier.config import ModelConfig
from minifrontier.evaluation.benchmark import (
    BenchmarkRecord,
    ComparisonKey,
    read_record,
    write_record,
)
from minifrontier.evaluation.code import (
    assert_no_contamination,
    dotnet_available,
    load_fixtures,
    normalized_hash,
    score_csharp,
    score_fixture_predictions,
    score_python,
)
from minifrontier.evaluation.fim import score_fim
from minifrontier.evaluation.language import MiniFrontierEvalLM, harness_settings
from minifrontier.evaluation.sft import score_sft_responses
from minifrontier.evaluation.validation import (
    ValidationBatch,
    batches_from_texts,
    evaluate_token_batches,
)
from minifrontier.model import MiniFrontier

ROOT = Path(__file__).parents[1]


class FixedLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, tokens: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.fixed_logits[: tokens.shape[0], : tokens.shape[1]])


def test_validation_metrics_match_hand_computation_and_ignore_padding() -> None:
    logits = torch.zeros(1, 4, 4)
    model = FixedLogitsModel(logits)
    batch = ValidationBatch(torch.tensor([[1, 2, 3, 0]]), utf8_bytes=2)
    metrics = evaluate_token_batches(model, [batch], pad_id=0)  # type: ignore[arg-type]
    assert metrics.predicted_tokens == 2
    assert metrics.cross_entropy == pytest.approx(math.log(4.0))
    assert metrics.perplexity == pytest.approx(4.0)
    assert metrics.bits_per_byte == pytest.approx(2.0)


def test_validation_rejects_truncation_that_would_bias_bpb(mini_tokenizer) -> None:
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        batches_from_texts(mini_tokenizer, ["long text " * 20], max_seq_len=3)


def test_code_and_fim_scoring_is_explicit_and_tested() -> None:
    source = "def add(a, b):\n    return a + b\n"
    tests = "assert add(2, 3) == 5"
    assert score_python(source).compiles
    with pytest.raises(ValueError, match="disabled"):
        score_python(source, tests=tests)
    assert score_python(source, tests=tests, execute_trusted_fixture=True).tests_passed
    broken = score_python("def broken(:\n")
    assert not broken.syntax_valid
    fim = score_fim(
        "def add(a, b):\n    ",
        "\n",
        "return a + b",
        "return a + b",
        tests=tests,
        execute_trusted_fixture=True,
    )
    assert fim.exact and fim.functional


@pytest.mark.skipif(not dotnet_available(), reason="dotnet SDK not found on PATH")
def test_score_csharp_compiles_a_real_method_wrapped_in_a_class() -> None:
    source = "int Add(int a, int b)\n{\n    return a + b;\n}\n"
    score = score_csharp(source)
    assert score.compiles and score.syntax_valid
    assert score.tests_passed is None  # C# scoring never executes, by design


@pytest.mark.skipif(not dotnet_available(), reason="dotnet SDK not found on PATH")
def test_score_csharp_rejects_a_real_broken_completion() -> None:
    broken = score_csharp("int Add(int a, int b)\n{\n    return a +\n}\n")
    assert not broken.compiles


@pytest.mark.skipif(not dotnet_available(), reason="dotnet SDK not found on PATH")
def test_score_fixture_predictions_dispatches_csharp_fixtures_by_language() -> None:
    fixtures = [
        {
            "id": "cs-1",
            "kind": "completion",
            "language": "csharp",
            "prompt": "int Add(int a, int b)\n{\n    ",
            "reference": "return a + b;\n}\n",
        }
    ]
    scored = score_fixture_predictions(fixtures, {"cs-1": "return a + b;\n}\n"})
    assert scored[0].exact and scored[0].compiles and scored[0].functional is None


def test_score_fixture_predictions_rejects_tests_on_csharp_fixtures() -> None:
    fixtures = [
        {
            "id": "cs-1",
            "kind": "completion",
            "language": "csharp",
            "prompt": "int Add(int a, int b)\n{\n    ",
            "reference": "return a + b;\n}\n",
            "tests": "assert Add(2, 3) == 5",
        }
    ]
    with pytest.raises(ValueError, match="does not support tests"):
        score_fixture_predictions(fixtures, {"cs-1": "return a + b;\n}\n"})


def test_score_fixture_predictions_rejects_unknown_language() -> None:
    fixtures = [
        {
            "id": "x-1",
            "kind": "completion",
            "language": "rust",
            "prompt": "fn add(",
            "reference": "a, b) -> i32 { a + b }",
        }
    ]
    with pytest.raises(ValueError, match="unknown language"):
        score_fixture_predictions(fixtures, {"x-1": "a, b) -> i32 { a + b }"})


@pytest.mark.skipif(not dotnet_available(), reason="dotnet SDK not found on PATH")
def test_real_csharp_fixture_file_round_trips_through_the_real_scorer() -> None:
    """The committed `eval/fixtures/code_csharp_fim_v1*.jsonl` pair, scored for
    real end to end -- the same "prove the scorer/report path" role the
    existing Python reference-predictions file already serves."""

    fixtures_path = ROOT / "eval" / "fixtures" / "code_csharp_fim_v1.jsonl"
    predictions_path = ROOT / "eval" / "fixtures" / "code_csharp_fim_v1_reference_predictions.jsonl"
    fixtures = load_fixtures(fixtures_path)
    predictions = {
        str(row["id"]): str(row["prediction"])
        for row in (json.loads(line) for line in predictions_path.read_text().splitlines() if line)
    }
    scores = score_fixture_predictions(fixtures, predictions)
    assert len(scores) == 5
    assert all(score.exact for score in scores)
    assert all(score.syntax_valid and score.compiles for score in scores)


def test_contamination_check_uses_normalized_hashes() -> None:
    fixtures = [{"id": "one", "prompt": "hello", "reference": "world"}]
    with pytest.raises(ValueError, match="contamination"):
        assert_no_contamination(fixtures, {normalized_hash("hello")})
    assert_no_contamination(fixtures, set())


def test_benchmark_schema_round_trip_and_comparability(tmp_path) -> None:
    key = ComparisonKey("data-v1", "abc", 10, 4, 8, "seed-1", "eval-v1")
    record = BenchmarkRecord("run-1", key, {"loss": 2.0}, 3.0, 4.0, 1.0, 0, 128, "cpu", [])
    path = tmp_path / "record.json"
    write_record(record, path)
    loaded = read_record(path)
    assert loaded == record
    assert loaded.comparable_to(record)
    changed = ComparisonKey("data-v2", "abc", 10, 4, 8, "seed-1", "eval-v1")
    assert not record.comparable_to(
        BenchmarkRecord("run-2", changed, {}, None, None, 0.0, 0, 0, "cpu", [])
    )


def test_lm_eval_adapter_local_smoke(mini_tokenizer) -> None:
    config = ModelConfig.tiny_edu(
        vocab_size=mini_tokenizer.vocab_size,
        max_seq_len=16,
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_ff=32,
    )
    adapter = MiniFrontierEvalLM(MiniFrontier(config), mini_tokenizer, max_gen_tokens=2)
    likelihood = adapter.loglikelihood([SimpleNamespace(args=("a", "b"))])
    rolling = adapter.loglikelihood_rolling([SimpleNamespace(args=("ab",))])
    generated = adapter.generate_until(
        [SimpleNamespace(args=("a", {"max_gen_toks": 1, "until": []}))]
    )
    assert len(likelihood) == len(rolling) == len(generated) == 1
    assert math.isfinite(likelihood[0][0]) and math.isfinite(rolling[0])
    settings = harness_settings()
    assert settings["tasks"] == ["arc_easy", "hellaswag", "piqa"]


def test_score_fast_path_matches_sliding_window_reference(mini_tokenizer) -> None:
    # MF-113: _score's fast single-pass path must agree with the original,
    # slower per-token sliding-window loop (kept as _score_sliding_window)
    # whenever no truncation is needed -- this is a pure performance change,
    # any numeric drift here would be a real correctness bug, not a win.
    config = ModelConfig.tiny_edu(
        vocab_size=mini_tokenizer.vocab_size,
        max_seq_len=64,
        n_layers=2,
        d_model=16,
        n_heads=2,
        d_ff=32,
    )
    adapter = MiniFrontierEvalLM(MiniFrontier(config), mini_tokenizer)
    prefix = mini_tokenizer.encode("the quick brown fox")
    continuation = mini_tokenizer.encode(" jumps over the lazy dog")
    fast_log_prob, fast_greedy = adapter._score(prefix, continuation)
    slow_log_prob, slow_greedy = adapter._score_sliding_window(prefix, continuation)
    assert fast_greedy == slow_greedy
    assert fast_log_prob == pytest.approx(slow_log_prob, abs=1e-4)


def test_score_empty_continuation_is_trivially_greedy(mini_tokenizer) -> None:
    config = ModelConfig.tiny_edu(
        vocab_size=mini_tokenizer.vocab_size,
        max_seq_len=16,
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_ff=32,
    )
    adapter = MiniFrontierEvalLM(MiniFrontier(config), mini_tokenizer)
    assert adapter._score(mini_tokenizer.encode("a"), []) == (0.0, True)


def test_score_falls_back_to_sliding_window_when_sequence_exceeds_max_length(
    mini_tokenizer,
) -> None:
    # A tiny max_seq_len forces (prefix + continuation) past max_length, which
    # must route to the slow-but-correct fallback rather than silently
    # truncating with the fast path's single fixed window.
    config = ModelConfig.tiny_edu(
        vocab_size=mini_tokenizer.vocab_size,
        max_seq_len=4,
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_ff=32,
    )
    adapter = MiniFrontierEvalLM(MiniFrontier(config), mini_tokenizer)
    prefix = mini_tokenizer.encode("the quick brown fox jumps")
    continuation = mini_tokenizer.encode(" over the lazy dog")
    log_prob, is_greedy = adapter._score(prefix, continuation)
    assert math.isfinite(log_prob)
    assert isinstance(is_greedy, bool)


def test_harness_settings_include_extended_adds_the_mf086_tasks() -> None:
    settings = harness_settings(include_extended=True)
    assert settings["tasks"] == [
        "arc_easy",
        "hellaswag",
        "piqa",
        "blimp",
        "lambada_openai",
        "winogrande",
        "openbookqa",
        "commonsense_qa",
        "boolq",
    ]


def test_harness_settings_include_gsm8k_and_extended_compose() -> None:
    settings = harness_settings(include_gsm8k=True, include_extended=True)
    assert settings["tasks"] == [
        "arc_easy",
        "hellaswag",
        "piqa",
        "gsm8k",
        "blimp",
        "lambada_openai",
        "winogrande",
        "openbookqa",
        "commonsense_qa",
        "boolq",
    ]


def test_sft_scoring_is_transparent_and_handles_missing_responses() -> None:
    prompts = [
        {"id": "one", "category": "instruction", "required_substrings": ["BLUE"]},
        {"id": "two", "category": "unknown", "required_substrings": []},
    ]
    result = score_sft_responses(prompts, {"one": "BLUE"})
    assert result["count"] == 2
    assert result["non_empty_rate"] == 0.5
    assert result["required_match_rate"] == 1.0
