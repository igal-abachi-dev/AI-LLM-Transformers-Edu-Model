import pytest
import torch

from minifrontier.cache import KVCache
from minifrontier.config import ModelConfig
from minifrontier.generation import generate
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.speculative_decoding import (
    SpeculativeStats,
    _speculative_append_is_safe,
    speculative_generate,
)


def _tiny_hybrid_model_and_heads(seed: int = 0) -> tuple[MiniFrontier, MTPHeads]:
    torch.manual_seed(seed)
    config = ModelConfig.tiny_modern(vocab_size=64, max_seq_len=64, local_window=8)
    model = MiniFrontier(config).eval()
    mtp_heads = MTPHeads(d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=1)
    return model, mtp_heads


def test_speculative_generate_matches_plain_greedy_decoding_across_a_ring_wrap() -> None:
    # local_window=8, so 20 new tokens spans well past the point a local ring
    # layer wraps -- the exact-match property must hold on both sides of that
    # boundary, not just before it.
    model, mtp_heads = _tiny_hybrid_model_and_heads()
    prompt = torch.tensor([[1, 2, 3, 4]])
    expected = generate(model, prompt.clone(), max_new_tokens=20, temperature=0)
    actual, stats = speculative_generate(model, mtp_heads, prompt.clone(), max_new_tokens=20)
    assert torch.equal(actual, expected)
    assert stats.proposed >= 0
    assert 0 <= stats.accepted <= stats.proposed


def test_speculative_generate_matches_plain_greedy_decoding_with_a_different_seed() -> None:
    model, mtp_heads = _tiny_hybrid_model_and_heads(seed=7)
    prompt = torch.tensor([[5, 6, 7]])
    expected = generate(model, prompt.clone(), max_new_tokens=12, temperature=0)
    actual, _ = speculative_generate(model, mtp_heads, prompt.clone(), max_new_tokens=12)
    assert torch.equal(actual, expected)


def test_speculative_generate_accept_and_reject_paths_are_both_exercised(monkeypatch) -> None:
    """Scripted, deterministic trace through both branches -- not left to chance."""

    model, mtp_heads = _tiny_hybrid_model_and_heads()
    scripted = iter([5, 7, 7, 9, 11, 13])

    def fake_greedy(logits: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[next(scripted)]])

    monkeypatch.setattr("minifrontier.speculative_decoding._greedy", fake_greedy)
    prompt = torch.tensor([[1, 2, 3]])
    tokens, stats = speculative_generate(model, mtp_heads, prompt, max_new_tokens=4)
    # call 1: real=5, call 2: draft=7, call 3: verdict=7 (== draft -> ACCEPT),
    # call 4: bonus=9, call 5: new draft=11, call 6: verdict=13 (!= 11 -> REJECT).
    assert tokens.tolist() == [[1, 2, 3, 5, 7, 9, 13]]
    assert stats.proposed == 2
    assert stats.accepted == 1
    assert stats.acceptance_rate == pytest.approx(0.5)


def test_speculative_generate_stops_at_eos_mid_accept(monkeypatch) -> None:
    model, mtp_heads = _tiny_hybrid_model_and_heads()
    scripted = iter([5, 2, 2, 9])  # draft (2) chosen to equal eos_id

    def fake_greedy(logits: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[next(scripted)]])

    monkeypatch.setattr("minifrontier.speculative_decoding._greedy", fake_greedy)
    prompt = torch.tensor([[1, 3, 4]])
    tokens, stats = speculative_generate(model, mtp_heads, prompt, max_new_tokens=5, eos_id=2)
    # real=5 (emit), verdict=2 == draft=2 -> accept, emit(2) is EOS -> stop
    # before the bonus token (9) is ever emitted.
    assert tokens.tolist() == [[1, 3, 4, 5, 2]]
    assert stats.accepted == 1


def test_speculative_generate_validates_arguments() -> None:
    model, mtp_heads = _tiny_hybrid_model_and_heads()
    with pytest.raises(ValueError, match="single stream"):
        speculative_generate(model, mtp_heads, torch.tensor([[1], [2]]), max_new_tokens=1)
    empty_prompt = torch.zeros((1, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="non-empty"):
        speculative_generate(model, mtp_heads, empty_prompt, max_new_tokens=1)
    with pytest.raises(ValueError, match="negative"):
        speculative_generate(model, mtp_heads, torch.tensor([[1]]), max_new_tokens=-1)
    with pytest.raises(ValueError, match="vocabulary"):
        speculative_generate(model, mtp_heads, torch.tensor([[1]]), max_new_tokens=1, eos_id=999)


def test_speculative_generate_zero_new_tokens_returns_prompt_unchanged() -> None:
    model, mtp_heads = _tiny_hybrid_model_and_heads()
    prompt = torch.tensor([[1, 2, 3]])
    tokens, stats = speculative_generate(model, mtp_heads, prompt, max_new_tokens=0)
    assert torch.equal(tokens, prompt)
    assert stats.proposed == 0
    assert stats.accepted == 0


def test_speculative_append_is_safe_becomes_false_exactly_at_the_ring_wrap_boundary() -> None:
    # local_window=4: safe while length+2 <= 4, unsafe once length+2 > 4. Every
    # layer must be appended to together (matching how model.forward always
    # extends a whole KVCache in lockstep) or cache.length desynchronizes.
    config = ModelConfig.tiny_modern(vocab_size=64, max_seq_len=64, local_window=4)
    cache = KVCache.allocate(
        config, batch_size=1, device=torch.device("cpu"), capacity=64, bounded_local=True
    )
    two_tokens = (
        torch.randn(1, config.n_kv_heads, 2, config.head_dim),
        torch.randn(1, config.n_kv_heads, 2, config.head_dim),
    )
    for layer in cache.layers:
        layer.append(*two_tokens, start_pos=0)
    assert cache.length == 2
    assert _speculative_append_is_safe(cache)  # 2 + 2 <= 4

    one_token = (
        torch.randn(1, config.n_kv_heads, 1, config.head_dim),
        torch.randn(1, config.n_kv_heads, 1, config.head_dim),
    )
    for layer in cache.layers:
        layer.append(*one_token, start_pos=2)
    assert cache.length == 3
    assert not _speculative_append_is_safe(cache)  # 3 + 2 > 4


def test_speculative_stats_acceptance_rate_handles_zero_proposed() -> None:
    assert SpeculativeStats(proposed=0, accepted=0).acceptance_rate == 0.0
