"""Multi-Token Prediction (MTP): an optional, off-by-default training-time auxiliary loss.

Beginner's map of this file
----------------------------
Normal training asks each position to predict only its immediate next token
(t+1). MTP adds extra, smaller prediction heads that also try to guess further
ahead (t+2, t+3, ...), each contributing its own loss term added to the main
one. The idea -- see Gloeckle et al., "Better & Faster Large Language Models
via Multi-token Prediction", and DeepSeek-V3's production use of it -- is that
predicting further ahead forces the model to build a sharper internal
representation sooner, which can improve how much a model learns per token.

This is a deliberately simplified variant. Each extra head here is a single
linear projection reading the SAME final hidden state the main next-token head
reads, not a separate small transformer block per depth the way DeepSeek-V3's
full design works. That keeps the whole mechanism visible in one short file, at
the cost of not reproducing DeepSeek-V3's exact architecture -- a fair trade
for a project whose entire point is "every important operation stays
explainable from the local source."

Why this lives outside ``MiniFrontier``
----------------------------------------
MTP heads are never part of the model class and never appear in its
``state_dict()``. They are extra parameters owned by whichever training script
uses them, feeding off ``MiniFrontier.forward(..., return_hidden_states=True)``
purely to compute an auxiliary loss. Configuration lives entirely in
``TrainingConfig`` (``mtp_extra_heads``, ``mtp_loss_weight`` -- see
``training.py``), never in ``ModelConfig``. This matters concretely:
``load_training_checkpoint`` (``checkpoint.py``) requires a checkpoint's saved
config to equal the current model's config *exactly*. If MTP added a field to
``ModelConfig``, every checkpoint saved before that field existed would fail to
load. Keeping MTP entirely out of the model and its config means a checkpoint's
``model.safetensors``/``config.json`` are byte-identical whether or not MTP was
used to train it -- every already-released MiniFrontier model stays exactly as
loadable as it is today.
"""

from __future__ import annotations

import torch
from torch import nn

from minifrontier.loss import next_token_loss_stats


class MTPHeads(nn.Module):
    """One extra linear head per additional predicted offset (t+2, t+3, ...).

    Reads the ``[B, S, d_model]`` hidden state ``MiniFrontier.forward`` exposes
    via ``return_hidden_states=True`` and produces ``n_extra_heads`` independent
    ``[B, S, vocab_size]`` logit tensors. Heads are untied from each other and
    from the model's own ``lm_head`` -- each is free to specialize for its own
    prediction distance.
    """

    def __init__(self, *, d_model: int, vocab_size: int, n_extra_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or vocab_size <= 0:
            raise ValueError("d_model and vocab_size must be positive")
        if n_extra_heads <= 0:
            raise ValueError("n_extra_heads must be positive")
        self.n_extra_heads = n_extra_heads
        self.heads = nn.ModuleList(
            nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_extra_heads)
        )
        # Same small-random-noise convention as MiniFrontier's own initializer
        # (model.py's `_initialize`), so an MTP head starts no more confidently
        # wrong than any other freshly built projection.
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=d_model**-0.5)

    def loss_sum_and_count(
        self,
        hidden_states: torch.Tensor,
        tokens: torch.Tensor,
        *,
        loss_mask: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed auxiliary loss and valid-target count across every extra head.

        Head ``i`` (0-indexed) predicts token ``t + 2 + i`` -- the main
        ``lm_head`` already owns ``t + 1``, so the first extra head starts two
        steps ahead. Returned summed rather than averaged, matching
        ``next_token_loss_stats``, so a caller combining this with the primary
        loss can divide once, exactly, instead of averaging two averages.
        """

        if hidden_states.ndim != 3 or tokens.ndim != 2:
            raise ValueError(
                "hidden_states must be [batch, sequence, d_model] and tokens [batch, sequence]"
            )
        if hidden_states.shape[:2] != tokens.shape:
            raise ValueError("hidden_states and tokens must share batch and sequence dimensions")

        total_loss = hidden_states.new_zeros(())
        total_count = torch.zeros((), dtype=torch.long, device=hidden_states.device)
        for extra_index, head in enumerate(self.heads):
            offset = 2 + extra_index
            if tokens.shape[1] < offset + 1:
                # This head's offset has no valid target in a sequence this
                # short. Contribute nothing rather than raising -- the main
                # head (offset=1) and any shorter-offset extra heads can still
                # train fine on the same batch.
                continue
            logits = head(hidden_states)
            loss_sum, count = next_token_loss_stats(
                logits,
                tokens,
                loss_mask=loss_mask,
                ignore_index=ignore_index,
                offset=offset,
            )
            total_loss = total_loss + loss_sum
            total_count = total_count + count
        return total_loss, total_count
