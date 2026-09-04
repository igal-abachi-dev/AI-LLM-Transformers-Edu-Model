import torch

from minifrontier.loss import next_token_loss_stats
from minifrontier.mtp import MTPHeads


def test_mtp_heads_rejects_non_positive_n_extra_heads() -> None:
    try:
        MTPHeads(d_model=8, vocab_size=16, n_extra_heads=0)
    except ValueError as error:
        assert "n_extra_heads" in str(error)
    else:
        raise AssertionError("expected a ValueError for n_extra_heads=0")


def test_mtp_heads_rejects_non_positive_dims() -> None:
    try:
        MTPHeads(d_model=0, vocab_size=16, n_extra_heads=1)
    except ValueError as error:
        assert "d_model" in str(error) or "vocab_size" in str(error)
    else:
        raise AssertionError("expected a ValueError for d_model=0")


def test_mtp_heads_produces_finite_loss_and_gradients() -> None:
    torch.manual_seed(3)
    mtp = MTPHeads(d_model=8, vocab_size=16, n_extra_heads=2)
    hidden_states = torch.randn(2, 10, 8, requires_grad=True)
    tokens = torch.randint(0, 16, (2, 10))
    loss_sum, count = mtp.loss_sum_and_count(hidden_states, tokens)
    assert torch.isfinite(loss_sum)
    assert count.item() > 0
    (loss_sum / count).backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for head in mtp.heads:
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()


def test_mtp_heads_offsets_match_manual_shift() -> None:
    """Head 0 must grade t+2, head 1 must grade t+3 -- exactly, not off by one."""

    torch.manual_seed(7)
    mtp = MTPHeads(d_model=6, vocab_size=12, n_extra_heads=2)
    hidden_states = torch.randn(1, 9, 6)
    tokens = torch.randint(0, 12, (1, 9))

    with torch.no_grad():
        logits_head0 = mtp.heads[0](hidden_states)
        logits_head1 = mtp.heads[1](hidden_states)
    expected_sum_0, expected_count_0 = next_token_loss_stats(logits_head0, tokens, offset=2)
    expected_sum_1, expected_count_1 = next_token_loss_stats(logits_head1, tokens, offset=3)

    with torch.no_grad():
        total_sum, total_count = mtp.loss_sum_and_count(hidden_states, tokens)
    assert torch.allclose(total_sum, expected_sum_0 + expected_sum_1)
    assert total_count == expected_count_0 + expected_count_1


def test_mtp_heads_skips_heads_with_no_valid_targets_in_short_sequence() -> None:
    """A 2-token sequence has a target for offset=2 (none) -- head 0 contributes
    nothing rather than raising, so the batch can still train the main head."""

    torch.manual_seed(11)
    mtp = MTPHeads(d_model=4, vocab_size=8, n_extra_heads=1)
    hidden_states = torch.randn(1, 2, 4)
    tokens = torch.randint(0, 8, (1, 2))
    loss_sum, count = mtp.loss_sum_and_count(hidden_states, tokens)
    assert loss_sum.item() == 0.0
    assert count.item() == 0
