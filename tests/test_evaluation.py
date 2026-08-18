from __future__ import annotations

import math
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
    normalized_hash,
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


def test_sft_scoring_is_transparent_and_handles_missing_responses() -> None:
    prompts = [
        {"id": "one", "category": "instruction", "required_substrings": ["BLUE"]},
        {"id": "two", "category": "unknown", "required_substrings": []},
    ]
    result = score_sft_responses(prompts, {"one": "BLUE"})
    assert result["count"] == 2
    assert result["non_empty_rate"] == 0.5
    assert result["required_match_rate"] == 1.0
