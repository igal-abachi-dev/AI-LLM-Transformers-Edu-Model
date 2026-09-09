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
