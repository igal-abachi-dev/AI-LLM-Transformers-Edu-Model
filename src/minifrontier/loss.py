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

from typing import Any

import torch
from torch.nn import functional as F


def next_token_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
    offset: int = 1,
) -> torch.Tensor:
    """Predict token ``t+offset`` from logits at ``t`` and average valid positions.

    Returns a single number: the mean surprise per scored token, in nats. This is
    the value ``.backward()`` is called on during training. ``offset`` defaults to
    the ordinary next-token case (``t+1``); a larger value is what Multi-Token
    Prediction's extra heads use to grade a further-ahead guess (see ``mtp.py``).
    """

    if logits.ndim != 3 or tokens.ndim != 2:
        raise ValueError("logits must be [batch, sequence, vocab] and tokens [batch, sequence]")
    if logits.shape[:2] != tokens.shape:
        raise ValueError("logits and tokens must share batch and sequence dimensions")
    if offset < 1:
        raise ValueError("offset must be at least 1")
    if tokens.shape[1] < offset + 1:
        raise ValueError("next-token loss requires at least offset + 1 tokens")
    if loss_mask is not None and loss_mask.shape != tokens.shape:
        raise ValueError("loss_mask must have the same shape as tokens")

    loss_sum, count = next_token_loss_stats(
        logits,
        tokens,
        loss_mask=loss_mask,
        ignore_index=ignore_index,
        offset=offset,
    )
    return loss_sum / count.clamp_min(1)


def next_token_loss_stats(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
    offset: int = 1,
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
    if offset < 1:
        raise ValueError("offset must be at least 1")
    if tokens.shape[1] < offset + 1:
        raise ValueError("next-token loss requires at least offset + 1 tokens")
    if loss_mask is not None and loss_mask.shape != tokens.shape:
        raise ValueError("loss_mask must have the same shape as tokens")

    # The shift, which trips up everyone the first time. Position t's scores are
    # graded against the token at t+offset, so we drop the last `offset` positions
    # (nothing that far ahead follows them) and the first `offset` tokens are never
    # a target (nothing that far back precedes them). Both become [B, S-offset].
    shifted_logits = logits[:, :-offset, :].contiguous()
    targets = tokens[:, offset:].contiguous()
    # cross_entropy wants a flat list of predictions, so [B, S-offset, V] ->
    # [B*(S-offset), V] and [B, S-offset] -> [B*(S-offset)]. reduction="none" keeps
    # one loss per token instead of averaging straight away, which is what lets us
    # mask afterwards.
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
        valid &= loss_mask[:, offset:].bool()
    count = valid.sum()
    # Multiplying by the boolean mask zeroes the skipped positions; they then
    # contribute nothing to either the sum or the count.
    return (per_token * valid).sum(), count


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Cross-entropy (plus optional z-loss) computed over ``lm_head``-sized
    chunks, never materializing the full ``[N, vocab_size]`` logits tensor.

    Why a custom ``autograd.Function`` rather than just looping and calling
    ``F.cross_entropy`` per chunk: a plain loop still needs one ``.backward()``
    per chunk to free that chunk's logits before the next one is built, and
    every one of those calls re-walks the *entire* upstream graph (the whole
    transformer) to reach ``hidden``'s parents -- turning one backward pass
    into ``n_chunks`` of them. Instead, ``forward`` here computes the
    gradient with respect to ``hidden`` and ``weight`` directly, chunk by
    chunk (closed-form: ``dL/dlogits = softmax(logits) - one_hot(target)``,
    the standard cross-entropy gradient), discards each chunk's ``[C,
    vocab_size]`` logits immediately, and accumulates only ``hidden``- and
    ``weight``-shaped gradients -- the same size as the tensors being
    differentiated, never vocab-sized. ``backward`` then does no
    recomputation at all; it just returns what ``forward`` already built,
    letting the rest of the model's backward pass run exactly once.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        chunk_size: int,
        z_loss_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_rows = hidden.shape[0]
        # FP32 (or better) throughout: the same "reductions stay FP32 under
        # autocast" convention the rest of the project follows (attention.py's
        # manual reference path, rope.py's rotation), written explicitly since a
        # hand-derived gradient gets no help from autocast's own rules. Never
        # downcasts an already-FP64 caller -- gradcheck (tests/test_loss.py)
        # needs that to hold, exactly like rope.apply_rotary.
        compute_dtype = hidden.dtype if hidden.dtype == torch.float64 else torch.float32
        grad_hidden_loss = torch.zeros_like(hidden, dtype=compute_dtype)
        grad_weight_loss = torch.zeros_like(weight, dtype=compute_dtype)
        grad_hidden_z = torch.zeros_like(hidden, dtype=compute_dtype)
        grad_weight_z = torch.zeros_like(weight, dtype=compute_dtype)
        loss_sum = hidden.new_zeros((), dtype=compute_dtype)
        z_loss_sum = hidden.new_zeros((), dtype=compute_dtype)

        weight_f32 = weight.to(compute_dtype)
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            hidden_chunk = hidden[start:end].to(compute_dtype)
            target_chunk = targets[start:end]
            valid_chunk = valid[start:end].to(compute_dtype)

            logits_chunk = hidden_chunk @ weight_f32.T
            log_sum_exp = torch.logsumexp(logits_chunk, dim=-1)
            target_logit = logits_chunk.gather(-1, target_chunk.unsqueeze(-1)).squeeze(-1)
            per_token_loss = (log_sum_exp - target_logit) * valid_chunk
            loss_sum += per_token_loss.sum()

            probabilities = torch.softmax(logits_chunk, dim=-1)
            grad_logits_loss = probabilities.clone()
            grad_logits_loss.scatter_add_(
                -1, target_chunk.unsqueeze(-1), -valid_chunk.unsqueeze(-1)
            )
            grad_logits_loss *= valid_chunk.unsqueeze(-1)
            grad_hidden_loss[start:end] = grad_logits_loss @ weight_f32
            grad_weight_loss += grad_logits_loss.T @ hidden_chunk

            if z_loss_weight:
                z_term = z_loss_weight * log_sum_exp.pow(2) * valid_chunk
                z_loss_sum += z_term.sum()
                # d(z_loss_weight * lse^2)/dlogits = 2 * z_loss_weight * lse * softmax(logits)
                grad_logits_z = (2.0 * z_loss_weight * log_sum_exp * valid_chunk).unsqueeze(
                    -1
                ) * probabilities
                grad_hidden_z[start:end] = grad_logits_z @ weight_f32
                grad_weight_z += grad_logits_z.T @ hidden_chunk

            del logits_chunk, probabilities, grad_logits_loss

        ctx.save_for_backward(grad_hidden_loss, grad_weight_loss, grad_hidden_z, grad_weight_z)
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        return loss_sum, z_loss_sum

    @staticmethod
    def backward(
        ctx: Any, grad_loss_sum: torch.Tensor, grad_z_loss_sum: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        grad_hidden_loss, grad_weight_loss, grad_hidden_z, grad_weight_z = ctx.saved_tensors
        # Composed via the actual incoming gradients for each output rather than
        # assuming the caller weights loss_sum/z_loss_sum equally -- correct
        # regardless of how they get combined downstream.
        grad_hidden = grad_loss_sum * grad_hidden_loss + grad_z_loss_sum * grad_hidden_z
        grad_weight = grad_loss_sum * grad_weight_loss + grad_z_loss_sum * grad_weight_z
        return (
            grad_hidden.to(ctx.hidden_dtype),
            grad_weight.to(ctx.weight_dtype),
            None,
            None,
            None,
            None,
        )


def chunked_next_token_loss_stats(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    tokens: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
    offset: int = 1,
    chunk_size: int = 1024,
    z_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-efficient equivalent of ``next_token_loss_stats`` plus z-loss.

    Takes the model's final hidden states and ``lm_head.weight`` directly,
    instead of already-materialized ``[B, S, vocab_size]`` logits, and never
    forms that full tensor -- see ``_ChunkedCrossEntropy`` for why this needs
    a custom autograd Function rather than just a loop. Returns
    ``(loss_sum, count, z_loss_sum)``: the first two are exactly what
    ``next_token_loss_stats`` returns (same shift/masking rules, same
    sum-not-mean convention for exact microbatch accumulation); ``z_loss_sum``
    is the summed PaLM-style ``z_loss_weight * log_sum_exp(logits)**2``
    stability penalty (zero when ``z_loss_weight == 0.0``), meant to be added
    directly to ``loss_sum`` before backpropagating -- it already has its
    weight baked in, unlike Multi-Token Prediction's separately-weighted
    auxiliary loss (see ``mtp.py``).
    """

    if hidden_states.ndim != 3 or tokens.ndim != 2:
        raise ValueError(
            "hidden_states must be [batch, sequence, d_model] and tokens [batch, sequence]"
        )
    if hidden_states.shape[:2] != tokens.shape:
        raise ValueError("hidden_states and tokens must share batch and sequence dimensions")
    if weight.ndim != 2 or weight.shape[1] != hidden_states.shape[-1]:
        raise ValueError("weight must be [vocab_size, d_model], matching hidden_states")
    if offset < 1:
        raise ValueError("offset must be at least 1")
    if tokens.shape[1] < offset + 1:
        raise ValueError("next-token loss requires at least offset + 1 tokens")
    if loss_mask is not None and loss_mask.shape != tokens.shape:
        raise ValueError("loss_mask must have the same shape as tokens")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if z_loss_weight < 0:
        raise ValueError("z_loss_weight must be non-negative")

    shifted_hidden = hidden_states[:, :-offset, :].contiguous()
    targets = tokens[:, offset:].contiguous()
    valid = targets.ne(ignore_index)
    if loss_mask is not None:
        valid &= loss_mask[:, offset:].bool()
    count = valid.sum()

    flat_hidden = shifted_hidden.reshape(-1, shifted_hidden.shape[-1])
    # ignore_index (-100) is not a valid gather/scatter index; safe wherever it
    # matters because `valid` (built above) zeroes that row's contribution to
    # both the loss and the gradient regardless of which target ID it holds.
    flat_targets = targets.reshape(-1).long().clamp(min=0, max=weight.shape[0] - 1)
    flat_valid = valid.reshape(-1)

    loss_sum, z_loss_sum = _ChunkedCrossEntropy.apply(
        flat_hidden, weight, flat_targets, flat_valid, chunk_size, z_loss_weight
    )
    return loss_sum, count, z_loss_sum
