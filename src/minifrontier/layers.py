"""Small neural-network primitives used by every MiniFrontier preset.

Beginner's map of this file
---------------------------
Two tiny building blocks. Every ``TransformerBlock`` uses ``RMSNorm`` twice and
``SwiGLU`` once, and neither of them ever looks at another token.

* ``RMSNorm`` is a volume knob. Numbers travelling up through a deep network
  tend to grow layer after layer until training explodes. RMSNorm rescales each
  token's vector back to a standard size, then multiplies by one learned number
  per slot so the model can still say "this feature matters more than that one".
* ``SwiGLU`` is the thinking room. Attention (the other half of a block) mixes
  information *between* tokens; SwiGLU processes each token *on its own*, first
  widening it into a bigger space to work in and then squeezing it back down.

Shapes never change across either one: ``[batch, sequence, d_model]`` goes in,
the same shape comes out. That is what lets you stack twenty of them.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization with one learned scale per feature.

    Plain version: take the ``dimension`` numbers describing one token, work out
    their typical size (the root mean square), divide everything by it, then
    multiply by a learned per-slot ``weight``. The direction the vector points in
    is untouched -- only its overall loudness is standardized.

    Compared with the older LayerNorm this skips subtracting the mean, which is
    one less pass over the data and works just as well in practice. That is why
    LLaMA-style models, and this one, use it everywhere.
    """

    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        # Starts at all ones, so an untrained RMSNorm is pure normalization and
        # the model has to learn any per-feature emphasis it wants.
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.weight.numel():
            raise ValueError(
                f"expected final dimension {self.weight.numel()}, got {inputs.shape[-1]}"
            )
        # One number per token: the average of its squared features. `.float()`
        # forces FP32 even when the model runs in BF16, because squaring and then
        # taking a reciprocal square root of small numbers loses accuracy fast.
        mean_square = inputs.float().pow(2).mean(-1, keepdim=True)
        # rsqrt(x) is 1/sqrt(x). `eps` keeps the division safe for an all-zero token.
        normalized = inputs.float() * torch.rsqrt(mean_square + self.eps)
        # Cast back so callers get the dtype they handed in: BF16 in, BF16 out.
        return (normalized * self.weight.float()).to(dtype=inputs.dtype)


class SwiGLU(nn.Module):
    """Dense gated feed-forward network: down(silu(gate(x)) * up(x)).

    Three matrices, applied to each token independently:

    * ``up_proj`` widens ``d_model`` -> ``d_ff`` and carries the content.
    * ``gate_proj`` widens the same input a second time, then passes through SiLU
      to become a dimmer switch that can turn each of the ``d_ff`` features up,
      down, or off.
    * ``down_proj`` multiplies the two together and squeezes back to ``d_model``.

    Read as English: "here is an idea" (``up``) times "how much does this idea
    apply to this token right now" (``gate``). The gate is what makes this better
    than a plain two-matrix MLP, and it is why most of a modern model's weights
    live in this class rather than in attention -- along with, it is thought, most
    of its raw factual knowledge.

    ``bias=False`` throughout: with an RMSNorm in front of every sublayer, biases
    cost parameters and buy essentially nothing.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Both projections read the SAME input; this is a fork, not a pipeline.
        # silu(x) = x * sigmoid(x): like ReLU but smooth, and slightly negative for
        # small negative x, which lets the gate subtract as well as pass through.
        # [B, S, d_model] -> [B, S, d_ff] -> [B, S, d_model]
        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))
