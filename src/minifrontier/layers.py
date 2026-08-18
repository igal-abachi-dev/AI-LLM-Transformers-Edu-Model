"""Small neural-network primitives used by every MiniFrontier preset."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization with one learned scale per feature."""

    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.weight.numel():
            raise ValueError(
                f"expected final dimension {self.weight.numel()}, got {inputs.shape[-1]}"
            )
        mean_square = inputs.float().pow(2).mean(-1, keepdim=True)
        normalized = inputs.float() * torch.rsqrt(mean_square + self.eps)
        return (normalized * self.weight.float()).to(dtype=inputs.dtype)


class SwiGLU(nn.Module):
    """Dense gated feed-forward network: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))
