import pytest
import torch

from minifrontier.cache import KVCache
from minifrontier.config import ModelConfig
from minifrontier.generation import generate
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.speculative_decoding import (
    SpeculativeStats,
    _rejection_sample,
    _speculative_append_is_safe,
    speculative_generate,
    speculative_generate_sampled,
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


# --- MF-118: exact rejection sampling for temperature-only sampling ---


def test_rejection_sample_reproduces_the_target_distribution_exactly() -> None:
    """The core distributional-equivalence proof, at the mechanism level, not
    through a full model -- so it can use arbitrary, deliberately mismatched
    target/draft distributions rather than whatever a random tiny transformer
    happens to produce. This is the real check MF-118's acceptance criterion
    asks for: a many-sample empirical comparison, not a single-sample match
    (sampling is inherently random, so bit-identity is the wrong bar here --
    see `speculative_generate`'s own greedy tests for that different bar).

    A chi-squared goodness-of-fit test against the *exact* target distribution
    `p` (not another empirical sample) removes double-sampling noise from one
    side of the comparison. `target` and `draft` are deliberately different
    random distributions -- if the accept/reject/resample math were wrong
    (e.g. resampling from raw `p` instead of the residual), the draft's own
    skew would leak into the output and this test would fail; a correct
    implementation reproduces `p` regardless of what `draft` looks like.
    """

    torch.manual_seed(0)
    vocab = 12
    target_probabilities = torch.softmax(torch.randn(1, vocab) * 2.0, dim=-1)
    draft_probabilities = torch.softmax(torch.randn(1, vocab) * 2.0, dim=-1)

    generator = torch.Generator().manual_seed(123)
    trials = 20_000
    counts = torch.zeros(vocab)
    for _ in range(trials):
        draft_token = torch.multinomial(draft_probabilities, num_samples=1, generator=generator)
        token, _accepted = _rejection_sample(
            target_probabilities, draft_token, draft_probabilities, generator=generator
        )
        counts[token.item()] += 1

    expected = target_probabilities.squeeze(0) * trials
    chi_squared = ((counts - expected) ** 2 / expected).sum().item()
    # 11 degrees of freedom (vocab - 1); the real critical value at p=0.001 is
    # ~31.26 -- 45 gives real margin against a flaky false failure while still
    # being a meaningful, real bound (a genuinely wrong resample formula
    # produces a statistic in the hundreds or more at this sample size, not a
    # borderline one).
    assert chi_squared < 45.0, f"chi-squared={chi_squared} -- output does not match target p"


def test_rejection_sample_always_accepts_when_draft_and_target_agree() -> None:
    # p == q everywhere: min(1, p/q) == 1 for whatever token got drafted, so
    # acceptance is certain regardless of the random draw -- a direct check of
    # the clamp(max=1.0) behavior, not just the general distributional test.
    probabilities = torch.softmax(torch.randn(1, 16), dim=-1)
    generator = torch.Generator().manual_seed(1)
    for _ in range(50):
        draft_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
        token, accepted = _rejection_sample(
            probabilities, draft_token, probabilities, generator=generator
        )
        assert accepted is True
        assert torch.equal(token, draft_token)


def _tiny_hybrid_model_and_heads_wide_vocab(seed: int = 0) -> tuple[MiniFrontier, MTPHeads]:
    # A wider vocabulary than the module-level helper's default 64 gives the
    # accept/reject test below more real opportunities to exercise both
    # branches within a short, fast generation.
    torch.manual_seed(seed)
    config = ModelConfig.tiny_modern(vocab_size=256, max_seq_len=128, local_window=16)
    model = MiniFrontier(config).eval()
    mtp_heads = MTPHeads(d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=1)
    return model, mtp_heads


def test_speculative_generate_sampled_runs_end_to_end_and_is_reproducible() -> None:
    model, mtp_heads = _tiny_hybrid_model_and_heads_wide_vocab()
    prompt = torch.tensor([[1, 2, 3, 4]])

    generator_a = torch.Generator().manual_seed(99)
    tokens_a, stats_a = speculative_generate_sampled(
        model, mtp_heads, prompt.clone(), max_new_tokens=24, temperature=0.8, generator=generator_a
    )
    generator_b = torch.Generator().manual_seed(99)
    tokens_b, stats_b = speculative_generate_sampled(
        model, mtp_heads, prompt.clone(), max_new_tokens=24, temperature=0.8, generator=generator_b
    )
    # Same seed, same everything else -> bit-identical output. Not the
    # exactness guarantee this technique is actually about (that's the
    # distributional test above), just ordinary, expected determinism.
    assert torch.equal(tokens_a, tokens_b)
    assert tokens_a.shape == (1, 4 + 24)
    # A real generation this long, on a hybrid model whose local layers never
    # wrap (max_seq_len=128 > 4+24), should genuinely exercise drafting.
    assert stats_a.proposed > 0
    assert stats_a == stats_b


def test_speculative_generate_sampled_validates_arguments() -> None:
    model, mtp_heads = _tiny_hybrid_model_and_heads_wide_vocab()
    with pytest.raises(ValueError, match="single stream"):
        speculative_generate_sampled(
            model, mtp_heads, torch.tensor([[1], [2]]), max_new_tokens=1, temperature=0.8
        )
    empty_prompt = torch.zeros((1, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="non-empty"):
        speculative_generate_sampled(
            model, mtp_heads, empty_prompt, max_new_tokens=1, temperature=0.8
        )
    with pytest.raises(ValueError, match="negative"):
        speculative_generate_sampled(
            model, mtp_heads, torch.tensor([[1]]), max_new_tokens=-1, temperature=0.8
        )
    with pytest.raises(ValueError, match="vocabulary"):
        speculative_generate_sampled(
            model, mtp_heads, torch.tensor([[1]]), max_new_tokens=1, temperature=0.8, eos_id=99999
        )
    with pytest.raises(ValueError, match="temperature > 0"):
        speculative_generate_sampled(
            model, mtp_heads, torch.tensor([[1]]), max_new_tokens=1, temperature=0.0
        )


def test_speculative_generate_sampled_zero_new_tokens_returns_prompt_unchanged() -> None:
    model, mtp_heads = _tiny_hybrid_model_and_heads_wide_vocab()
    prompt = torch.tensor([[1, 2, 3]])
    tokens, stats = speculative_generate_sampled(
        model, mtp_heads, prompt, max_new_tokens=0, temperature=0.8
    )
    assert torch.equal(tokens, prompt)
    assert stats.proposed == 0
    assert stats.accepted == 0
