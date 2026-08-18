"""Rotary position embeddings using the split-half LLaMA convention."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    """Map ``[x1, x2]`` to ``[-x2, x1]`` along the final dimension."""

    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    inputs: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply precomputed ``[sequence, head_dim]`` cosine/sine values."""

    if inputs.ndim != 4:
        raise ValueError("rotary inputs must be [batch, heads, sequence, head_dim]")
    if cosine.shape != sine.shape or cosine.shape != inputs.shape[-2:]:
        raise ValueError("cosine and sine must match [sequence, head_dim]")
    cosine = cosine.to(dtype=inputs.dtype).unsqueeze(0).unsqueeze(0)
    sine = sine.to(dtype=inputs.dtype).unsqueeze(0).unsqueeze(0)
    return inputs * cosine + rotate_half(inputs) * sine


class RoPE(nn.Module):
    """Generate rotary cosine/sine tables for explicit token positions."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even number")
        if max_seq_len <= 0 or theta <= 0:
            raise ValueError("max_seq_len and theta must be positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be a one-dimensional integer tensor")
        if positions.numel() == 0:
            raise ValueError("positions cannot be empty")
        frequencies = torch.outer(
            positions.to(device=device, dtype=torch.float32),
            self.inverse_frequency.to(device=device),
        )
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)
