"""Rotary position embeddings using the split-half LLaMA convention.

Beginner's map of this file
---------------------------
Attention on its own has no sense of order: it would score "dog bites man"
exactly like "man bites dog". RoPE fixes that by *rotating* each token's query
and key vectors by an angle proportional to that token's position.

Why a rotation rather than "add the position number in"? Because when two tokens
are later compared with a dot product, their two rotations partly cancel and what
survives depends on the *difference* between them -- how far apart the tokens are.
So the model learns about relative distance, which generalizes much better than
memorizing absolute slot numbers.

Three facts worth carrying around:

* RoPE is applied to Q and K only, never to V. It decides *where to look*, not
  *what to fetch*.
* It has no learned parameters at all. The tables are pure arithmetic.
* A head's ``head_dim`` numbers are treated as ``head_dim / 2`` little 2-D arrows,
  each with its own rotation speed. The fast arrows encode "a few tokens apart";
  the slow ones encode "hundreds of tokens apart".
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    """Map ``[x1, x2]`` to ``[-x2, x1]`` along the final dimension.

    This is the "turn every arrow 90 degrees" half of the rotation formula.

    Note the pairing convention: feature ``i`` is paired with feature
    ``i + head_dim/2`` (split-half, LLaMA style), not with its neighbour ``i + 1``
    (interleaved, as in the original RoPE paper). Both are valid rotations, but a
    model trained under one convention produces nonsense when read under the
    other -- which is exactly why ``tests/test_rope.py`` checks this against an
    independent implementation instead of only against itself.
    """

    # chunk(2) splits [..., D] into two halves of D/2: the "x" and "y" components
    # of every arrow. Swapping them and negating one is a quarter turn.
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    inputs: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply precomputed ``[sequence, head_dim]`` cosine/sine values.

    Rotating a 2-D arrow ``(x1, x2)`` by angle ``a`` gives
    ``(x1*cos a - x2*sin a, x2*cos a + x1*sin a)``. Written for the whole vector at
    once that is exactly ``x * cos + rotate_half(x) * sin``, which is the single
    line at the bottom of this function.

    ``inputs`` is ``[batch, heads, sequence, head_dim]`` -- either Q or K, after
    the heads have been split out.
    """

    if inputs.ndim != 4:
        raise ValueError("rotary inputs must be [batch, heads, sequence, head_dim]")
    if cosine.shape != sine.shape or cosine.shape != inputs.shape[-2:]:
        raise ValueError("cosine and sine must match [sequence, head_dim]")
    # Every batch item and every head is rotated identically, so add two leading
    # size-1 axes and let broadcasting do the work: [S, D] -> [1, 1, S, D].
    cosine = cosine.to(dtype=inputs.dtype).unsqueeze(0).unsqueeze(0)
    sine = sine.to(dtype=inputs.dtype).unsqueeze(0).unsqueeze(0)
    return inputs * cosine + rotate_half(inputs) * sine


class RoPE(nn.Module):
    """Generate rotary cosine/sine tables for explicit token positions.

    This is an ``nn.Module`` only so the frequency table travels with
    ``model.to(device)``; it holds no learned weights. ``MiniFrontier.forward``
    calls it once per forward pass and hands the same two tables to every layer,
    because the rotation depends on position alone and not on the layer.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even number")
        if max_seq_len <= 0 or theta <= 0:
            raise ValueError("max_seq_len and theta must be positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # One rotation speed per arrow: 1 / theta**(2i/head_dim) for i = 0, 1, 2...
        # Arrow 0 advances a full radian per token, so it comes back around every
        # six or so -- useful for "is this token right next to me?". The slowest
        # arrow advances only about 1/theta of a radian per token, so it barely
        # moves across the whole context and encodes coarse, long-range distance.
        # Raising `theta` slows every arrow down, which is the standard knob for
        # stretching an already-trained model to longer contexts.
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # A buffer, not a Parameter: it moves with the model but is never trained.
        # persistent=False keeps it out of checkpoints since it is recomputable.
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
        # `positions` holds ABSOLUTE token indices, not 0..S-1 offsets. When
        # generating with a KV cache the model sees one token per call, and that
        # token must still be rotated as position 501 rather than position 0.
        # outer product -> angles[p, i] = position p * speed i, in FP32 for accuracy.
        frequencies = torch.outer(
            positions.to(device=device, dtype=torch.float32),
            self.inverse_frequency.to(device=device),
        )
        # Duplicate the half-width table so it lines up with the split-half pairing
        # used by `rotate_half`: feature i and feature i + head_dim/2 are the two
        # components of one arrow, so they share one angle. -> [S, head_dim]
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)
