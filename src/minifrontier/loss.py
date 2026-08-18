"""Language-model losses with explicit token shifting and masking."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def next_token_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Predict token ``t+1`` from logits at ``t`` and average valid positions."""

    if logits.ndim != 3 or tokens.ndim != 2:
        raise ValueError("logits must be [batch, sequence, vocab] and tokens [batch, sequence]")
    if logits.shape[:2] != tokens.shape:
        raise ValueError("logits and tokens must share batch and sequence dimensions")
    if tokens.shape[1] < 2:
        raise ValueError("next-token loss requires at least two tokens")
    if loss_mask is not None and loss_mask.shape != tokens.shape:
        raise ValueError("loss_mask must have the same shape as tokens")

    loss_sum, count = next_token_loss_stats(
        logits,
        tokens,
        loss_mask=loss_mask,
        ignore_index=ignore_index,
    )
    return loss_sum / count.clamp_min(1)


def next_token_loss_stats(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed next-token loss and valid-target count for exact accumulation."""

    if logits.ndim != 3 or tokens.ndim != 2:
        raise ValueError("logits must be [batch, sequence, vocab] and tokens [batch, sequence]")
    if logits.shape[:2] != tokens.shape:
        raise ValueError("logits and tokens must share batch and sequence dimensions")
    if tokens.shape[1] < 2:
        raise ValueError("next-token loss requires at least two tokens")
    if loss_mask is not None and loss_mask.shape != tokens.shape:
        raise ValueError("loss_mask must have the same shape as tokens")

    shifted_logits = logits[:, :-1, :].contiguous()
    targets = tokens[:, 1:].contiguous()
    per_token = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(targets)
    valid = targets.ne(ignore_index)
    if loss_mask is not None:
        valid &= loss_mask[:, 1:].bool()
    count = valid.sum()
    return (per_token * valid).sum(), count
