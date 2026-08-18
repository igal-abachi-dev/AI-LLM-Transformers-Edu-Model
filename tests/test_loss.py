import torch
from torch.nn import functional as F

from minifrontier.loss import next_token_loss


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
