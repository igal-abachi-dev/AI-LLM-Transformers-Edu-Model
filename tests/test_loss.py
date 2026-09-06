import pytest
import torch
from torch.nn import functional as F

from minifrontier.loss import chunked_next_token_loss_stats, next_token_loss, next_token_loss_stats


def test_next_token_loss_shifts_in_the_correct_direction() -> None:
    tokens = torch.tensor([[1, 2, 3]])
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = 5
    logits[0, 1, 3] = 5
    expected = F.cross_entropy(logits[:, :2].reshape(-1, 5), tokens[:, 1:].reshape(-1))
    assert torch.allclose(next_token_loss(logits, tokens), expected)


def test_loss_mask_excludes_prompt_positions() -> None:
    tokens = torch.tensor([[1, 2, 3, 4]])
    logits = torch.randn(1, 4, 8, requires_grad=True)
    mask = torch.tensor([[False, False, True, True]])
    loss = next_token_loss(logits, tokens, loss_mask=mask)
    expected = F.cross_entropy(logits[:, 1:3].reshape(-1, 8), tokens[:, 2:4].reshape(-1))
    assert torch.allclose(loss, expected)


def test_all_masked_loss_is_safe_differentiable_zero() -> None:
    logits = torch.randn(1, 3, 8, requires_grad=True)
    tokens = torch.tensor([[1, 2, 3]])
    loss = next_token_loss(logits, tokens, loss_mask=torch.zeros_like(tokens, dtype=torch.bool))
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.count_nonzero() == 0


def test_offset_two_shifts_two_steps_ahead() -> None:
    tokens = torch.tensor([[1, 2, 3, 4]])
    logits = torch.randn(1, 4, 6)
    loss = next_token_loss(logits, tokens, offset=2)
    # Position 0's score is graded against token[2], position 1's against
    # token[3]. Positions 2 and 3 have nothing two steps ahead of them, so they
    # drop out of both the logits and the targets.
    expected = F.cross_entropy(logits[:, :2].reshape(-1, 6), tokens[:, 2:].reshape(-1))
    assert torch.allclose(loss, expected)


def test_offset_below_one_is_rejected() -> None:
    tokens = torch.tensor([[1, 2, 3]])
    logits = torch.randn(1, 3, 5)
    try:
        next_token_loss(logits, tokens, offset=0)
    except ValueError as error:
        assert "offset" in str(error)
    else:
        raise AssertionError("expected a ValueError for offset < 1")


def test_offset_leaving_no_targets_is_rejected() -> None:
    tokens = torch.tensor([[1, 2, 3]])
    logits = torch.randn(1, 3, 5)
    try:
        next_token_loss(logits, tokens, offset=3)
    except ValueError as error:
        assert "offset" in str(error)
    else:
        raise AssertionError("expected a ValueError when offset leaves no valid targets")


def _reference_loss_sum_and_count(hidden, weight, tokens, **kwargs):
    """Full-materialization reference: exactly what `chunked_next_token_loss_stats`
    is supposed to match, computed the ordinary (memory-heavy) way."""

    logits = hidden.double() @ weight.double().T
    return next_token_loss_stats(logits, tokens, **kwargs)


def test_chunked_matches_unchunked_loss_sum_and_count() -> None:
    torch.manual_seed(0)
    batch, seq, d_model, vocab = 3, 7, 6, 11
    hidden = torch.randn(batch, seq, d_model, dtype=torch.float64)
    weight = torch.randn(vocab, d_model, dtype=torch.float64)
    tokens = torch.randint(0, vocab, (batch, seq))
    tokens[0, 3] = -100  # a real ignore_index target, not just an edge case
    loss_mask = torch.ones(batch, seq, dtype=torch.bool)
    loss_mask[1, :2] = False

    for offset in (1, 2):
        for chunk_size in (1, 4, 5, 1000):  # uneven, exact, and single-chunk sizes
            expected_sum, expected_count = _reference_loss_sum_and_count(
                hidden, weight, tokens, loss_mask=loss_mask, offset=offset
            )
            actual_sum, actual_count, z_loss_sum = chunked_next_token_loss_stats(
                hidden, weight, tokens, loss_mask=loss_mask, offset=offset, chunk_size=chunk_size
            )
            assert actual_count == expected_count
            assert torch.allclose(actual_sum, expected_sum, atol=1e-9)
            assert z_loss_sum.item() == 0.0


def test_chunked_gradients_match_unchunked_gradients() -> None:
    torch.manual_seed(1)
    batch, seq, d_model, vocab = 2, 6, 5, 9
    tokens = torch.randint(0, vocab, (batch, seq))
    base_hidden = torch.randn(batch, seq, d_model, dtype=torch.float64)
    base_weight = torch.randn(vocab, d_model, dtype=torch.float64)

    reference_hidden = base_hidden.clone().requires_grad_(True)
    reference_weight = base_weight.clone().requires_grad_(True)
    ref_logits = reference_hidden @ reference_weight.T
    ref_sum, ref_count = next_token_loss_stats(ref_logits, tokens)
    (ref_sum / ref_count).backward()

    chunked_hidden = base_hidden.clone().requires_grad_(True)
    chunked_weight = base_weight.clone().requires_grad_(True)
    chunk_sum, chunk_count, _ = chunked_next_token_loss_stats(
        chunked_hidden, chunked_weight, tokens, chunk_size=2
    )
    (chunk_sum / chunk_count).backward()

    assert torch.allclose(chunked_hidden.grad, reference_hidden.grad, atol=1e-9)
    assert torch.allclose(chunked_weight.grad, reference_weight.grad, atol=1e-9)


def test_chunked_cross_entropy_passes_gradcheck() -> None:
    """The definitive check for a hand-derived backward: compare the analytical
    gradient against finite differences, not just against another implementation
    that could share the same mistake."""

    torch.manual_seed(2)
    batch, seq, d_model, vocab = 2, 5, 4, 6
    hidden = torch.randn(batch, seq, d_model, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(vocab, d_model, dtype=torch.float64, requires_grad=True)
    tokens = torch.randint(0, vocab, (batch, seq))
    loss_mask = torch.randint(0, 2, (batch, seq)).bool()
    loss_mask[:, 0] = True  # guarantee at least one valid target after the shift

    def loss_only(h, w):
        loss_sum, count, _ = chunked_next_token_loss_stats(
            h, w, tokens, loss_mask=loss_mask, chunk_size=2
        )
        return loss_sum / count.clamp_min(1)

    assert torch.autograd.gradcheck(loss_only, (hidden, weight), eps=1e-6, atol=1e-4)


def test_z_loss_gradient_matches_plain_autograd_reference() -> None:
    torch.manual_seed(3)
    batch, seq, d_model, vocab = 2, 5, 4, 7
    tokens = torch.randint(0, vocab, (batch, seq))
    base_hidden = torch.randn(batch, seq, d_model, dtype=torch.float64)
    base_weight = torch.randn(vocab, d_model, dtype=torch.float64)
    z_weight = 0.1

    reference_hidden = base_hidden.clone().requires_grad_(True)
    reference_weight = base_weight.clone().requires_grad_(True)
    full_logits = reference_hidden @ reference_weight.T
    main_sum, count = next_token_loss_stats(full_logits, tokens)
    # z-loss is computed over the same shifted positions the main loss scores --
    # no loss_mask/ignore_index in this test, so every shifted position is valid.
    lse = full_logits[:, :-1, :].logsumexp(-1)
    z_sum = (z_weight * lse.pow(2)).sum()
    ((main_sum + z_sum) / count).backward()

    chunked_hidden = base_hidden.clone().requires_grad_(True)
    chunked_weight = base_weight.clone().requires_grad_(True)
    chunk_main, chunk_count, chunk_z = chunked_next_token_loss_stats(
        chunked_hidden, chunked_weight, tokens, chunk_size=2, z_loss_weight=z_weight
    )
    ((chunk_main + chunk_z) / chunk_count).backward()

    assert not torch.equal(chunk_z, torch.zeros_like(chunk_z))
    assert torch.allclose(chunked_hidden.grad, reference_hidden.grad, atol=1e-9)
    assert torch.allclose(chunked_weight.grad, reference_weight.grad, atol=1e-9)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@requires_cuda
def test_cuda_chunked_matches_unchunked_loss_under_real_autocast() -> None:
    """The gap the CPU/FP64 tests above cannot see: under a real
    torch.autocast(dtype=float16) context -- the precision every actual
    training run uses -- matmul casts its inputs to FP16 regardless of the
    explicit FP32 casts inside _ChunkedCrossEntropy (verified empirically:
    an FP32 tensor multiplied under autocast(float16) produces an FP16
    result). This is not a bug -- the unfused path's own `lm_head` matmul has
    the exact same property under the same context -- but nothing before this
    test actually exercised the combination, so a real divergence here would
    have gone uncaught."""

    device = torch.device("cuda")
    torch.manual_seed(6)
    batch, seq, d_model, vocab = 2, 6, 16, 64
    hidden = torch.randn(batch, seq, d_model, device=device)
    weight = torch.randn(vocab, d_model, device=device)
    tokens = torch.randint(0, vocab, (batch, seq), device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        reference_hidden = hidden.clone().requires_grad_(True)
        reference_weight = weight.clone().requires_grad_(True)
        logits = reference_hidden @ reference_weight.T
        ref_sum, ref_count = next_token_loss_stats(logits, tokens)
        (ref_sum / ref_count).backward()

        chunked_hidden = hidden.clone().requires_grad_(True)
        chunked_weight = weight.clone().requires_grad_(True)
        chunk_sum, chunk_count, _ = chunked_next_token_loss_stats(
            chunked_hidden, chunked_weight, tokens, chunk_size=4
        )
        (chunk_sum / chunk_count).backward()

    assert torch.isfinite(chunk_sum)
    assert torch.allclose(chunk_sum, ref_sum, atol=5e-2, rtol=5e-2)
    assert torch.allclose(chunked_hidden.grad, reference_hidden.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(chunked_weight.grad, reference_weight.grad, atol=5e-2, rtol=5e-2)


def test_chunked_loss_rejects_invalid_arguments() -> None:
    hidden = torch.randn(1, 4, 3)
    weight = torch.randn(5, 3)
    tokens = torch.randint(0, 5, (1, 4))
    try:
        chunked_next_token_loss_stats(hidden, weight, tokens, chunk_size=0)
    except ValueError as error:
        assert "chunk_size" in str(error)
    else:
        raise AssertionError("expected a ValueError for chunk_size=0")
    try:
        chunked_next_token_loss_stats(hidden, weight, tokens, z_loss_weight=-1.0)
    except ValueError as error:
        assert "z_loss_weight" in str(error)
    else:
        raise AssertionError("expected a ValueError for negative z_loss_weight")
    try:
        chunked_next_token_loss_stats(hidden, torch.randn(5, 4), tokens)
    except ValueError as error:
        assert "weight" in str(error)
    else:
        raise AssertionError("expected a ValueError for a d_model-mismatched weight")
