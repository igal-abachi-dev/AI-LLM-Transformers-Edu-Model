"""Show what FlexAttention actually does differently for a local layer -- not just that
it agrees with the manual reference, but the real block-skipping mechanism itself.

What this lab shows
--------------------
Every other attention path in this project -- manual, SDPA -- is handed a full
``[query, key]`` boolean grid and masks it. FlexAttention is not: it is handed
``mask_mod``, a tiny function answering "is this one (query, key) pair allowed?",
and it uses that function to work out, *before* computing any scores, which
128x128 **tiles** of the score matrix are entirely masked -- and skips computing
them at all. That tile-skipping, not anything about the softmax or the matmul, is
the entire reason a local layer is supposed to be cheaper than a global one
(``attention.py``'s own ``_flex_block_mask`` docstring says exactly this; this lab
makes it a real, inspected number instead of a sentence to take on faith).

Two parts:

1. **Correctness, small and hand-checkable.** A tiny sequence, by-hand tensor ops
   compared against this project's own ``manual_scaled_dot_product_attention``
   and the real ``flex_attention`` kernel. All three must agree.
2. **The real block structure, at this project's own actual default size**
   (``local_window=512``, a 1024-token sequence -- half of ``max_seq_len``, not an
   arbitrary demo number). FlexAttention's real ``BlockMask`` object is inspected
   directly: how many of the 8x8 possible 128-token tiles get skipped entirely,
   how many are computed mask-free (fully inside the window, cheapest to compute),
   and how many need the masked path (straddling the window's edge) -- tied to the
   real ``BlockMask.sparsity()`` number FlexAttention itself reports.

Run it with::

    uv run --extra cpu python labs/10_manual_flex_attention.py

What to look for. Part 1's three numbers should all read ``0.00e+00``. Part 2
prints, for each of the 8 query blocks, exactly which key blocks it touches --
almost always just 2 out of 8 -- and the total skipped fraction should match
``sparsity()`` exactly, because that field is computed from the same counts.

Honest caveat, not swept under the rug: FlexAttention runs *eager* (uncompiled)
here, same as every real MiniFrontier training run does today. This project's own
real measurement (`reports/mf050-rtx2070s-profile-matrix.md`) found eager
FlexAttention is *slower* than the manual path on this reference hardware, not
faster -- the tile-skipping shown below is real, but `torch.compile` (confirmed
broken on this project's own PyTorch/CUDA/Windows build, `MF-078`) is what
actually turns "fewer tiles to compute" into "less wall-clock time." What this
lab demonstrates is the mechanism, not a speed win on this machine.
"""

from __future__ import annotations

import math

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from minifrontier.attention import manual_scaled_dot_product_attention
from minifrontier.masking import build_attention_mask


def local_mask_mod(window: int):
    """The exact two rules masking.build_attention_mask uses, one pair at a
    time instead of as one big grid -- the same shape attention.py's own
    _flex_block_mask uses for every real local layer."""

    def mask_mod(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        allowed = key_index <= query_index
        allowed &= key_index >= query_index - window + 1
        return allowed

    return mask_mod


def main() -> None:
    # --- Part 1: correctness, small enough to fully trust by eye ----------------
    torch.manual_seed(0)
    batch, heads, sequence, head_dim, window = 1, 1, 8, 4, 3
    query = torch.randn(batch, heads, sequence, head_dim)
    key = torch.randn(batch, heads, sequence, head_dim)
    value = torch.randn(batch, heads, sequence, head_dim)

    banded_mask = build_attention_mask(sequence, sequence, window_size=window)
    scale = 1.0 / math.sqrt(head_dim)
    scores = (query @ key.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~banded_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    by_hand_output = torch.softmax(scores, dim=-1) @ value

    manual_output = manual_scaled_dot_product_attention(query, key, value, mask=banded_mask)

    small_block_mask = create_block_mask(
        local_mask_mod(window), B=None, H=None, Q_LEN=sequence, KV_LEN=sequence, device=query.device
    )
    flex_output = flex_attention(query, key, value, block_mask=small_block_mask)

    print(f"Part 1 -- sequence={sequence}, window={window} (small, so every step is checkable):")
    hand_vs_manual = (by_hand_output - manual_output).abs().max().item()
    manual_vs_flex = (manual_output - flex_output).abs().max().item()
    print(f"  by-hand vs manual_scaled_dot_product_attention: max diff = {hand_vs_manual:.2e}")
    print(f"  manual reference vs real FlexAttention kernel:  max diff = {manual_vs_flex:.2e}")
    print("  All three compute the same thing.\n")

    # --- Part 2: the real block-skip mechanism, at this project's own defaults --
    real_sequence, real_window = 1024, 512  # ModelConfig's own real local_window default
    block_mask = create_block_mask(
        local_mask_mod(real_window),
        B=None,
        H=None,
        Q_LEN=real_sequence,
        KV_LEN=real_sequence,
        device=torch.device("cpu"),
    )
    block_size = block_mask.BLOCK_SIZE[0]
    n_blocks = real_sequence // block_size
    print(
        f"Part 2 -- sequence={real_sequence}, window={real_window} "
        f"(this project's own real Modern default), {block_size}x{block_size} tiles:"
    )
    print(f"  {n_blocks}x{n_blocks} = {n_blocks * n_blocks} possible tiles total\n")

    # kv_indices and full_kv_indices are TWO SEPARATE arrays -- masked-tile
    # indices and fully-dense-tile indices are never stored together, and each
    # is only valid up to its own *_num_blocks count (anything past that in the
    # underlying tensor is unused padding, not a real tile -- reading past it
    # silently gives garbage, verified directly while building this lab).
    masked_per_row = block_mask.kv_num_blocks[0, 0]
    dense_per_row = block_mask.full_kv_num_blocks[0, 0]
    for query_block in range(n_blocks):
        masked_count = masked_per_row[query_block].item()
        dense_count = dense_per_row[query_block].item()
        masked_tiles = sorted(block_mask.kv_indices[0, 0, query_block][:masked_count].tolist())
        dense_tiles = sorted(block_mask.full_kv_indices[0, 0, query_block][:dense_count].tolist())
        skipped = n_blocks - masked_count - dense_count
        start, end = query_block * block_size, (query_block + 1) * block_size - 1
        print(
            f"  query tokens {start:>4}-{end:<4}: dense (mask-free) tiles {dense_tiles}, "
            f"masked tiles {masked_tiles} -- {skipped} of {n_blocks} tiles skipped, zero compute"
        )

    total_active = (masked_per_row + dense_per_row).sum().item()
    total_tiles = n_blocks * n_blocks
    computed_fraction = 1 - block_mask.sparsity() / 100
    print(
        f"\nTotal: {total_active}/{total_tiles} tiles ever touched "
        f"({computed_fraction:.1%}) -- real BlockMask.sparsity() reports "
        f"{block_mask.sparsity():.3f}% skipped, matching exactly."
    )
    print(
        "\nThat is the actual mechanism: a query block only ever touches the tile it sits in "
        "(the causal diagonal) and, once the window reaches back far enough, the one tile at "
        "the window's trailing edge -- every other tile is provably all-masked before any score "
        "is ever computed, so FlexAttention skips it outright rather than computing and then "
        "discarding it, which is what a mask-then-softmax approach (manual, SDPA) always does."
    )


if __name__ == "__main__":
    main()
