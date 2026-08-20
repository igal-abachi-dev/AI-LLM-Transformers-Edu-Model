"""Explicit attention masks shared by teaching and optimized paths.

Beginner's map of this file
---------------------------
A mask is a grid of yes/no answers to one question: "may query token *i* look at
key token *j*?". Exactly two rules are encoded here, and every attention variant
in this repository is one of them or both:

1. **Causal** -- ``key <= query``. A token may look at itself and everything
   before it, never at the future. Without this the model could cheat at
   next-token prediction by reading the answer, and would learn nothing useful.
   Drawn as a grid, it is the filled lower-left triangle.
2. **Sliding window** (local layers only) -- additionally
   ``key >= query - window + 1``. The token also forgets anything older than
   ``window`` positions, turning the triangle into a diagonal band.

``True`` means allowed. The fast kernels do not consume this tensor -- SDPA is
told ``is_causal=True`` and FlexAttention rebuilds the same two predicates inside
a ``mask_mod`` closure -- but they must agree with it, and the tests check that
they do. This file is the readable definition the rest of the code is measured
against.
"""

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
    """Return a boolean ``[query, key]`` mask where ``True`` means allowed.

    The ``*_start`` arguments exist for KV-cached generation, where the rows and
    columns of the grid are no longer numbered from zero. Decoding token 700 means
    one query row at absolute position 700 against 700 stored key columns; and a
    local layer's ring cache may only still hold positions 189..700, so the key
    axis needs its own origin.

    Everything below compares absolute positions, which is why the same function
    serves both a fresh prompt and a single cached decode step.
    """

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

    query_positions = query_start + torch.arange(query_length, device=device)  # [Sq]
    key_positions = key_start + torch.arange(key_length, device=device)  # [Sk]
    # Broadcasting [1, Sk] <= [Sq, 1] compares every pair at once and produces the
    # [Sq, Sk] causal triangle. No loops, no Python-level indexing.
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if window_size is not None:
        # Oldest key each query may still see. The `+ 1` makes the window inclusive
        # of the query itself, so window_size=1 would mean "only yourself".
        first_visible = query_positions.unsqueeze(1) - window_size + 1
        mask &= key_positions.unsqueeze(0) >= first_visible
    return mask
