from __future__ import annotations

import torch

from minifrontier.chat import complete_text
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads


def _tiny_model(mini_tokenizer, *, local_window: int = 8) -> MiniFrontier:
    torch.manual_seed(0)
    config = ModelConfig.tiny_modern(
        vocab_size=mini_tokenizer.vocab_size, max_seq_len=64, local_window=local_window
    )
    return MiniFrontier(config).eval()


def test_complete_text_without_mtp_heads_matches_previous_behavior(mini_tokenizer) -> None:
    model = _tiny_model(mini_tokenizer)
    result = complete_text(model, mini_tokenizer, "hello", max_new_tokens=5, seed=1)
    assert isinstance(result, str)


def test_complete_text_with_mtp_heads_greedy_matches_plain_decode(mini_tokenizer) -> None:
    """MF-105: speculative decoding is an exact acceleration of greedy decode --
    complete_text's output must be identical with or without mtp_heads."""

    model = _tiny_model(mini_tokenizer)
    plain = complete_text(model, mini_tokenizer, "hello world", max_new_tokens=10, seed=1)

    torch.manual_seed(0)
    mtp_heads = MTPHeads(
        d_model=model.config.d_model, vocab_size=model.config.vocab_size, n_extra_heads=1
    )
    accelerated = complete_text(
        model, mini_tokenizer, "hello world", max_new_tokens=10, seed=1, mtp_heads=mtp_heads
    )
    assert accelerated == plain


def test_complete_text_falls_back_to_plain_decode_under_sampling(
    mini_tokenizer, monkeypatch
) -> None:
    """A non-greedy request (temperature/top_k/top_p set) must never take the
    speculative path -- its exactness guarantee does not cover sampling."""

    model = _tiny_model(mini_tokenizer)
    torch.manual_seed(0)
    mtp_heads = MTPHeads(
        d_model=model.config.d_model, vocab_size=model.config.vocab_size, n_extra_heads=1
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("speculative_generate must not be called under sampling")

    monkeypatch.setattr("minifrontier.chat.speculative_generate", _fail_if_called)
    result = complete_text(
        model,
        mini_tokenizer,
        "hello",
        max_new_tokens=5,
        seed=1,
        temperature=0.8,
        mtp_heads=mtp_heads,
    )
    assert isinstance(result, str)


def test_complete_text_falls_back_to_plain_decode_under_repetition_penalty(
    mini_tokenizer, monkeypatch
) -> None:
    """MF-149: repetition_penalty/no_repeat_ngram_size are applied *before* the
    greedy argmax (sample_next_token), so -- like non-default temperature/top_k/
    top_p -- they can change which token greedy picks and must also fall back to
    plain decoding rather than silently taking the (no-longer-exact) speculative
    path."""

    model = _tiny_model(mini_tokenizer)
    torch.manual_seed(0)
    mtp_heads = MTPHeads(
        d_model=model.config.d_model, vocab_size=model.config.vocab_size, n_extra_heads=1
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("speculative_generate must not be called under repetition_penalty")

    monkeypatch.setattr("minifrontier.chat.speculative_generate", _fail_if_called)
    result = complete_text(
        model,
        mini_tokenizer,
        "hello",
        max_new_tokens=5,
        seed=1,
        repetition_penalty=1.2,
        mtp_heads=mtp_heads,
    )
    assert isinstance(result, str)


def test_complete_text_min_p_does_not_disable_speculative_decoding(mini_tokenizer) -> None:
    """min_p is a no-op under greedy decoding (sample_next_token returns via its
    temperature==0 early return before min_p is ever applied), so unlike
    repetition_penalty/no_repeat_ngram_size it must NOT force the plain-decode
    fallback -- speculative decoding stays exact and available."""

    model = _tiny_model(mini_tokenizer)
    torch.manual_seed(0)
    mtp_heads = MTPHeads(
        d_model=model.config.d_model, vocab_size=model.config.vocab_size, n_extra_heads=1
    )
    plain = complete_text(model, mini_tokenizer, "hello world", max_new_tokens=10, seed=1)
    accelerated = complete_text(
        model,
        mini_tokenizer,
        "hello world",
        max_new_tokens=10,
        seed=1,
        min_p=0.1,
        mtp_heads=mtp_heads,
    )
    assert accelerated == plain


def test_complete_text_repetition_penalty_reaches_generate(mini_tokenizer, monkeypatch) -> None:
    """The new keyword actually reaches model.generate(...), not just parsed and
    dropped."""

    model = _tiny_model(mini_tokenizer)
    captured = {}
    real_generate = model.generate

    def _capturing_generate(*args, **kwargs):
        captured.update(kwargs)
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(model, "generate", _capturing_generate)
    complete_text(
        model,
        mini_tokenizer,
        "hello",
        max_new_tokens=3,
        seed=1,
        min_p=0.05,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )
    assert captured["min_p"] == 0.05
    assert captured["repetition_penalty"] == 1.15
    assert captured["no_repeat_ngram_size"] == 3
