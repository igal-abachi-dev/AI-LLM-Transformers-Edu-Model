"""Explicit attention masks shared by teaching and optimized paths."""

from __future__ import annotations

import torch


def build_attention_mask(
    query_length: int,
    key_length: int,
    *,
    query_start: int = 0,
    key_start: int = 0,
    window_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a boolean ``[query, key]`` mask where ``True`` means allowed."""

    if query_length <= 0 or key_length <= 0:
        raise ValueError("query_length and key_length must be positive")
    if query_start < 0 or key_start < 0:
        raise ValueError("query_start and key_start cannot be negative")
    if key_start > query_start:
        raise ValueError("available keys cannot start after the first query")
    if query_start + query_length > key_start + key_length:
        raise ValueError("query positions cannot extend beyond available keys")
    if window_size is not None and window_size <= 0:
        raise ValueError("window_size must be positive when provided")

    query_positions = query_start + torch.arange(query_length, device=device)
    key_positions = key_start + torch.arange(key_length, device=device)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if window_size is not None:
        first_visible = query_positions.unsqueeze(1) - window_size + 1
        mask &= key_positions.unsqueeze(0) >= first_visible
    return mask
