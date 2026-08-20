"""Language-model losses with explicit token shifting and masking.

Beginner's map of this file
---------------------------
The training signal in one sentence: **every position must predict its
right-hand neighbour**. Feed in a 1,024-token sequence and one forward pass
yields 1,023 independent training examples, which is the only reason pretraining
is affordable at all.

The measurement is cross-entropy, best thought of as a surprise-o-meter. If the
model gave the true next token a 90% chance the loss is small (about 0.1 nats);
if it gave it 2% the loss is large (about 3.9). Averaged over positions, that
number is what backpropagation pushes downward.

Two independent ways to skip a position:

* ``ignore_index=-100`` inside ``labels`` -- PyTorch's standard convention for
  "there is no right answer here" (padding, truncated turns).
* ``loss_mask`` -- used by SFT so the model is graded only on the assistant's
  tokens and not on the user's.
"""

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
    """Predict token ``t+1`` from logits at ``t`` and average valid positions.

    Returns a single number: the mean surprise per scored token, in nats. This is
    the value ``.backward()`` is called on during training.
    """

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
    """Return summed next-token loss and valid-target count for exact accumulation.

    Same computation as ``next_token_loss``, but stopping one step short of the
    division. The trainer needs that: when a large batch is split into several
    microbatches, averaging each one and then averaging the averages silently
    over-weights the microbatches with fewer real tokens. Summing here and
    dividing once at the end is exact.
    """

    if logits.ndim != 3 or tokens.ndim != 2:
        raise ValueError("logits must be [batch, sequence, vocab] and tokens [batch, sequence]")
    if logits.shape[:2] != tokens.shape:
        raise ValueError("logits and tokens must share batch and sequence dimensions")
    if tokens.shape[1] < 2:
        raise ValueError("next-token loss requires at least two tokens")
    if loss_mask is not None and loss_mask.shape != tokens.shape:
        raise ValueError("loss_mask must have the same shape as tokens")

    # The shift, which trips up everyone the first time. Position t's scores are
    # graded against the token at t+1, so we drop the last position (nothing
    # follows it) and the first token is never a target (nothing precedes it).
    # Both become [B, S-1].
    shifted_logits = logits[:, :-1, :].contiguous()
    targets = tokens[:, 1:].contiguous()
    # cross_entropy wants a flat list of predictions, so [B, S-1, V] -> [B*(S-1), V]
    # and [B, S-1] -> [B*(S-1)]. reduction="none" keeps one loss per token instead
    # of averaging straight away, which is what lets us mask afterwards.
    per_token = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(targets)
    # Which positions actually count. `loss_mask` is sliced the same way as the
    # targets so the two stay aligned after the shift.
    valid = targets.ne(ignore_index)
    if loss_mask is not None:
        valid &= loss_mask[:, 1:].bool()
    count = valid.sum()
    # Multiplying by the boolean mask zeroes the skipped positions; they then
    # contribute nothing to either the sum or the count.
    return (per_token * valid).sum(), count
