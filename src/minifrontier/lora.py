"""From-scratch LoRA (Low-Rank Adaptation, Hu et al., arXiv:2106.09685) adapters.

Beginner's map of this file
----------------------------
Full fine-tuning updates every weight in a linear layer -- for a ``d_model x
d_ff`` matrix, that is a lot of numbers an optimizer has to carry its own
running-average state for (AdamW alone doubles or triples that memory).
LoRA's idea: freeze the original weight entirely, and add a small, separate
update ``B @ A`` alongside it, where ``A`` is ``rank x in_features`` and
``B`` is ``out_features x rank`` with ``rank`` far smaller than either
dimension. Only ``A`` and ``B`` ever get a gradient; the frozen weight never
moves at all::

    output = base(x) + scale * B @ (A @ x)

``A`` starts at a small random value, but ``B`` starts at exactly zero -- so
the very first forward pass through a freshly-wrapped layer is numerically
identical to the unwrapped base layer, not merely an approximation of one.
That zero-init is the one detail that is easy to get wrong and breaks
everything subtly (a nonzero-start adapter changes the model's behavior
before it has learned anything) if copied carelessly.

This project reimplements LoRA from scratch in plain PyTorch rather than
importing PEFT, matching every other technique here (Muon, cautious AdamW,
FlexAttention) -- ``AGENTS.md`` excludes Transformers Trainer/PEFT/TRL/
Accelerate from the core. Real, disclosed scope limit (``MF-168``): QLoRA's
own 4-bit quantized-base-weight storage is a separate, much larger piece of
engineering (a real NF4 quantization scheme, dequantize-on-the-fly forward
pass) and is not built here -- this module freezes the base weight at full
precision, which is the right tradeoff for this project's own real use case
(specializing an already-small MiniFrontier checkpoint, or fine-tuning a
teacher candidate that already fits this project's own reference hardware
without quantization), not for adapting a frozen weight too large to fit
memory any other way.
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen base ``nn.Linear`` plus a trainable rank-``rank`` adapter."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if base.bias is not None:
            # Real, disclosed scope limit, not an oversight: every nn.Linear
            # in this project's own real model is bias=False (an RMSNorm
            # sits in front of each sublayer, which makes a bias pointless --
            # see layers.py's own convention), so a biased base layer is
            # never actually exercised anywhere in this codebase.
            raise ValueError("LoRALinear does not support a biased base layer")
        self.base = base
        self.base.weight.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scale = self.alpha / self.rank
        # A: small random init (the same fan-in-aware init nn.Linear itself
        # uses for its own weight, so the adapter starts at a sane scale
        # relative to the layer it is attached to, not an arbitrary one).
        self.lora_down = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        # B: exactly zero. This is the one detail that matters most -- with
        # B=0, `scale * B @ (A @ x)` is the zero tensor regardless of what A
        # is, so a freshly-wrapped layer's forward pass is byte-identical to
        # the unwrapped base layer until real training actually moves B away
        # from zero. Initializing both A and B randomly (an easy mistake to
        # copy from a careless reading of the paper) would instead start
        # every wrapped layer with an arbitrary, untrained perturbation.
        self.lora_up = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapted = self.dropout(x) @ self.lora_down.T @ self.lora_up.T
        return base_out + self.scale * adapted

    def merged_weight(self) -> torch.Tensor:
        """The single, real combined weight this adapter is equivalent to --
        ``W_0 + scale * B @ A`` -- for exporting a merged, adapter-free
        release once training is done (the standard final step: an adapter
        left unmerged adds real inference overhead on every forward call)."""

        return self.base.weight + self.scale * (self.lora_up @ self.lora_down)

    def trainable_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return (self.lora_down, self.lora_up)


def apply_lora(
    model: nn.Module,
    *,
    target_pattern: str,
    rank: int,
    alpha: float | None = None,
    dropout: float = 0.0,
) -> list[str]:
    """Wrap every ``nn.Linear`` submodule whose dotted name matches
    ``target_pattern`` (a regex, e.g. ``r"(q_proj|k_proj|v_proj|out_proj)$"``
    for MiniFrontier's own real attention projections) with a ``LoRALinear``,
    in place. Returns the real list of wrapped module names, so a caller can
    confirm exactly what was (and was not) touched rather than guessing from
    the pattern alone.

    Real, disclosed interaction worth knowing, not a bug: MiniFrontier's own
    ``lm_head`` is tied to ``token_embedding`` by default (``tie_embeddings``)
    -- the same literal tensor object under two names. Matching ``lm_head``
    in ``target_pattern`` would freeze that shared tensor's gradient via this
    function's own ``requires_grad_(False)``, which also freezes
    ``token_embedding``'s own training since they are the same Parameter.
    Not specially handled here -- real target modules for specializing a
    model (attention/FFN projections) do not include the tied embedding
    pair, so this is deliberately left as a known interaction rather than a
    speculative special case for a target nothing here actually uses.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear) or not re.search(target_pattern, name):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        wrapped.append(name)
    return wrapped


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Every real trainable LoRA parameter in the model -- what an optimizer
    should actually be given once LoRA is applied, rather than a naive
    ``model.parameters()`` pass, which would also include every *other*,
    unwrapped layer's own normally-trainable-by-default weights."""

    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend(module.trainable_parameters())
    return params


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze every parameter in ``model`` except LoRA adapters -- the real,
    standard LoRA training setup (only ``A``/``B`` train; everything else,
    including any unwrapped layer, stays frozen). Call this *after*
    ``apply_lora`` so the adapters it just added are the ones left trainable.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in lora_parameters(model):
        parameter.requires_grad_(True)


def trainable_parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Real ``(trainable, total)`` parameter counts -- the concrete number
    LoRA exists to shrink. Counts each real storage tensor once (an
    ``id()``-keyed pass), matching ``MiniFrontier.parameter_count``'s own
    tied-weight-aware convention elsewhere in this project, so a tied
    embedding/lm_head pair is not double-counted into ``total``."""

    seen: set[int] = set()
    trainable = 0
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total
