import re

import pytest
import torch

from minifrontier.config import ModelConfig
from minifrontier.evaluation.retrieval import (
    NEEDLE_TEMPLATE,
    QUESTION,
    build_needle_prompt,
    run_needle_haystack_eval,
    run_needle_trial,
)
from minifrontier.model import MiniFrontier


def test_build_needle_prompt_respects_context_length_and_needle_position(mini_tokenizer) -> None:
    prompt_ids = build_needle_prompt(
        mini_tokenizer, context_length=200, needle_fraction=0.5, code="123456"
    )
    assert len(prompt_ids) == 200
    # The question must be the literal tail of the prompt.
    question_ids = mini_tokenizer.encode(QUESTION)
    assert prompt_ids[-len(question_ids) :] == question_ids
    # The needle text must actually be present somewhere in the decoded prompt.
    decoded = mini_tokenizer.decode(prompt_ids, skip_special_tokens=True)
    assert "123456" in decoded


def test_build_needle_prompt_at_the_two_extremes(mini_tokenizer) -> None:
    early = build_needle_prompt(mini_tokenizer, context_length=300, needle_fraction=0.0, code="1")
    late = build_needle_prompt(mini_tokenizer, context_length=300, needle_fraction=1.0, code="1")
    needle_ids = mini_tokenizer.encode(NEEDLE_TEMPLATE.format(code="1"))
    question_ids = mini_tokenizer.encode(QUESTION)
    # fraction=0.0: the needle is the very first thing in the prompt.
    assert early[: len(needle_ids)] == needle_ids
    # fraction=1.0: the needle sits immediately before the fixed question tail.
    assert late[-(len(question_ids) + len(needle_ids)) : -len(question_ids)] == needle_ids


def test_build_needle_prompt_rejects_invalid_arguments(mini_tokenizer) -> None:
    with pytest.raises(ValueError, match="needle_fraction"):
        build_needle_prompt(mini_tokenizer, context_length=100, needle_fraction=1.5, code="1")
    with pytest.raises(ValueError, match="context_length"):
        build_needle_prompt(mini_tokenizer, context_length=0, needle_fraction=0.5, code="1")
    with pytest.raises(ValueError, match="too short"):
        build_needle_prompt(mini_tokenizer, context_length=1, needle_fraction=0.5, code="1")


def test_run_needle_trial_reports_not_found_for_a_wrong_completion(
    monkeypatch, mini_tokenizer
) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(320, mini_tokenizer.vocab_size), max_seq_len=512)
    model = MiniFrontier(config).eval()

    def fake_generate(model, prompt, *, max_new_tokens, temperature):
        # Echo back a fixed, wrong continuation regardless of the real needle.
        wrong_ids = mini_tokenizer.encode(" 000000")
        tail = torch.tensor([wrong_ids], dtype=torch.long, device=prompt.device)
        return torch.cat([prompt, tail], dim=1)

    monkeypatch.setattr("minifrontier.evaluation.retrieval.generate", fake_generate)
    trial = run_needle_trial(model, mini_tokenizer, context_length=200, needle_fraction=0.3, seed=0)
    assert trial.code != "000000"  # the real random code, confirmed distinct from the wrong guess
    assert trial.found is False


def test_run_needle_trial_reports_found_when_the_real_code_is_echoed(
    monkeypatch, mini_tokenizer
) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(320, mini_tokenizer.vocab_size), max_seq_len=512)
    model = MiniFrontier(config).eval()
    captured_code = {}

    def fake_generate(model, prompt, *, max_new_tokens, temperature):
        # Can't know the code in advance here, so decode it back out of the
        # prompt itself (the real needle sentence is really embedded in it).
        prompt_text = mini_tokenizer.decode(prompt[0].tolist(), skip_special_tokens=True)
        match = re.search(r"secret code for today is (\d{6})", prompt_text)
        assert match is not None
        captured_code["value"] = match.group(1)
        tail_ids = mini_tokenizer.encode(f" {match.group(1)} is the code")
        tail = torch.tensor([tail_ids], dtype=torch.long, device=prompt.device)
        return torch.cat([prompt, tail], dim=1)

    monkeypatch.setattr("minifrontier.evaluation.retrieval.generate", fake_generate)
    trial = run_needle_trial(model, mini_tokenizer, context_length=200, needle_fraction=0.5, seed=1)
    assert trial.found is True
    assert trial.code == captured_code["value"]


def test_run_needle_haystack_eval_covers_the_full_grid(monkeypatch, mini_tokenizer) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(320, mini_tokenizer.vocab_size), max_seq_len=512)
    model = MiniFrontier(config).eval()

    def fake_generate(model, prompt, *, max_new_tokens, temperature):
        tail = torch.zeros((prompt.shape[0], 3), dtype=torch.long, device=prompt.device)
        return torch.cat([prompt, tail], dim=1)

    monkeypatch.setattr("minifrontier.evaluation.retrieval.generate", fake_generate)
    trials = run_needle_haystack_eval(
        model,
        mini_tokenizer,
        context_lengths=[150, 250],
        needle_fractions=[0.0, 0.5, 1.0],
        seed=42,
    )
    assert len(trials) == 6
    pairs = {(t.context_length, t.needle_fraction) for t in trials}
    assert pairs == {(150, 0.0), (150, 0.5), (150, 1.0), (250, 0.0), (250, 0.5), (250, 1.0)}


def test_run_needle_haystack_eval_rejects_empty_grids(mini_tokenizer) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(320, mini_tokenizer.vocab_size))
    model = MiniFrontier(config).eval()
    with pytest.raises(ValueError, match="non-empty"):
        run_needle_haystack_eval(model, mini_tokenizer, context_lengths=[], needle_fractions=[0.5])
