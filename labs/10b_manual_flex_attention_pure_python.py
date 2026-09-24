"""FlexAttention's real block-skipping mechanism with zero external libraries.

What this lab shows
--------------------
`10_manual_flex_attention.py` derives local attention by hand, then inspects the
real `torch.nn.attention.flex_attention.BlockMask` object to see which 128-token
tiles get skipped entirely. This lab goes one level further down, the same way
`00b_attention_pure_python.py` goes one level further than `00_attention_math.py`:
every operation -- matmul, softmax, masking, and this time the tile classification
itself -- is written out as plain Python lists, loops, and range comparisons.
Nothing here is a library trick; it is the literal logic those libraries exist to
speed up.

Two things, matching `10`'s own two parts exactly, with no PyTorch anywhere:

1. **The attention math.** Local attention needs no new computation, only a
   different mask -- confirmed again here, in pure Python this time, using
   `00b`'s own three building blocks (`transpose_2d`, `matmul_2d`, `softmax_1d`,
   restated below for this file to stand alone) unchanged.
2. **The tile classification.** `attention.py`'s own `_flex_block_mask` docstring
   says FlexAttention "works out which tiles of the score matrix are entirely
   masked so it can skip them completely." That is not a black box -- it is a
   small, checkable loop: for every (query-tile, key-tile) pair, look at the mask
   function's answer for every position inside that tile. All `True` -> dense
   (compute with no masking overhead). All `False` -> skip (zero compute). Mixed
   -> masked (compute, then mask). This function does exactly that, brute force,
   and at this project's own real `local_window=512`/1024-token default it
   reproduces the *exact* real counts `10` read out of the real `BlockMask`
   object -- 30 of 64 tiles ever touched, 53.125% skipped -- without importing
   torch at all.

Run it with::

    uv run --extra cpu python labs/10b_manual_flex_attention_pure_python.py

What to look for. Part 1 reuses `00b`'s own three-token Query/Key/Value numbers,
so it is directly comparable: under a plain causal mask token 2 blends all three
values, but give it `window=2` instead and token 0's value silently drops out of
the blend entirely, even though token 0 is still in the past. Part 2's printed
per-tile breakdown should read identically to `10`'s real PyTorch output.
"""

from __future__ import annotations

import math
from collections.abc import Callable

Vector1D = list[float]
Matrix2D = list[list[float]]
Mask2D = list[list[bool]]


# =====================================================================
# 1. The same three building blocks 00b_attention_pure_python.py derives --
#    restated here, unchanged, so this file has no import on another lab.
# =====================================================================


def transpose_2d(matrix: Matrix2D) -> Matrix2D:
    """[Rows, Cols] -> [Cols, Rows]. Equivalent to ATen's `tensor.transpose(-2, -1)`."""
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


def matmul_2d(a: Matrix2D, b: Matrix2D) -> Matrix2D:
    """[M, K] x [K, N] -> [M, N]. Equivalent to cuBLAS `gemm` / `torch.matmul`."""
    m, k, k_b, n = len(a), len(a[0]), len(b), len(b[0])
    if k_b != k:
        raise ValueError(f"Inner dimension mismatch: A is [{m}, {k}] but B is [{k_b}, {n}]")
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            result[i][j] = sum(a[i][x] * b[x][j] for x in range(k))
    return result


def softmax_1d(row: Vector1D) -> Vector1D:
    """Numerically stable softmax. Equivalent to ATen `SoftMaxKernel.cpp`."""
    max_val = max(row)
    exps = [0.0 if x == float("-inf") else math.exp(x - max_val) for x in row]
    total = sum(exps)
    if total == 0.0:
        raise ValueError("Sum of exponents is zero in softmax")
    return [e / total for e in exps]


# =====================================================================
# 2. The mask itself -- causal, plus one more rule for a local layer.
# =====================================================================


def local_mask_mod(query_index: int, key_index: int, *, window: int | None) -> bool:
    """The exact two rules masking.build_attention_mask uses, one pair at a time
    -- the same shape attention.py's own _flex_block_mask hands to real
    FlexAttention, and the only function Part 2 below ever calls."""

    if key_index > query_index:
        return False  # causal: no peeking at the future
    # local: also forgets anything older than `window` positions.
    return window is None or key_index >= query_index - window + 1


def banded_mask_pure_python(sequence: int, *, window: int | None) -> Mask2D:
    return [[local_mask_mod(q, k, window=window) for k in range(sequence)] for q in range(sequence)]


def manual_attention_pure_python(
    query: Matrix2D, key: Matrix2D, value: Matrix2D, *, mask: Mask2D
) -> Matrix2D:
    """Single-head attention, [Sq, D]/[Sk, D] in, [Sq, D] out -- the same four
    steps 00b's own manual_scaled_dot_product_attention_pure_python performs,
    trimmed to one head/batch since that is all this lab needs."""

    scale = 1.0 / math.sqrt(len(query[0]))
    raw_scores = matmul_2d(query, transpose_2d(key))
    masked_scores = [
        [raw_scores[i][j] * scale if mask[i][j] else float("-inf") for j in range(len(key))]
        for i in range(len(query))
    ]
    probabilities = [softmax_1d(row) for row in masked_scores]
    return matmul_2d(probabilities, value)


# =====================================================================
# 3. The block classification FlexAttention itself performs.
# =====================================================================


def classify_tile(
    query_start: int,
    key_start: int,
    block_size: int,
    mask_mod: Callable[[int, int], bool],
) -> str:
    """One (query-tile, key-tile) pair -> "dense", "masked", or "skip".

    Brute force, on purpose: check every one of the up to block_size^2 (query,
    key) pairs inside this one tile. Real FlexAttention does the equivalent
    check once, ahead of time, while building the BlockMask -- not per training
    step -- so this cost is paid once per (shape, window) combination, not once
    per forward pass, the same real reason attention.py's own `_BLOCK_MASK_CACHE`
    exists.
    """

    any_allowed = False
    any_denied = False
    for q in range(query_start, query_start + block_size):
        for k in range(key_start, key_start + block_size):
            if mask_mod(q, k):
                any_allowed = True
            else:
                any_denied = True
            if any_allowed and any_denied:
                return "masked"
    return "dense" if any_allowed else "skip"


def classify_all_tiles(
    sequence: int, block_size: int, mask_mod: Callable[[int, int], bool]
) -> list[dict[str, list[int]]]:
    """Per query-block: which key-blocks are dense, masked, or skipped."""

    n_blocks = sequence // block_size
    rows = []
    for query_block in range(n_blocks):
        dense: list[int] = []
        masked: list[int] = []
        for key_block in range(n_blocks):
            q_start, k_start = query_block * block_size, key_block * block_size
            kind = classify_tile(q_start, k_start, block_size, mask_mod)
            if kind == "dense":
                dense.append(key_block)
            elif kind == "masked":
                masked.append(key_block)
        rows.append({"dense": dense, "masked": masked})
    return rows


def main() -> None:
    # --- Part 1: same three-token story 00b tells, now with a window --------
    query = key = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    value = [[10.0, 0.0], [0.0, 20.0], [30.0, 30.0]]

    print("Part 1 -- same Q/K/V as 00b_attention_pure_python.py's own example:")
    print(f"  Query=Key={query}\n  Value={value}\n")

    causal_only = banded_mask_pure_python(3, window=None)
    causal_output = manual_attention_pure_python(query, key, value, mask=causal_only)
    print(f"  Plain causal, token 2's output: {[round(v, 4) for v in causal_output[2]]}")
    print("    (a real blend of all three Values -- token 2 can see everything before it)")

    windowed = banded_mask_pure_python(3, window=2)
    windowed_output = manual_attention_pure_python(query, key, value, mask=windowed)
    print(f"  window=2,      token 2's output: {[round(v, 4) for v in windowed_output[2]]}")
    print(
        "    (Value 0 -- [10.0, 0.0] -- has silently dropped out of the blend entirely: "
        "token 0 is 2 steps back, outside a window of 2, even though it is still in the past)\n"
    )

    # --- Part 2: the real block-skip mechanism, at this project's own defaults
    real_sequence, real_window, block_size = 1024, 512, 128
    n_blocks = real_sequence // block_size
    print(
        f"Part 2 -- sequence={real_sequence}, window={real_window} "
        f"(this project's own real Modern default), {block_size}x{block_size} tiles:"
    )
    print(f"  {n_blocks}x{n_blocks} = {n_blocks * n_blocks} possible tiles total\n")

    def mask_mod(q: int, k: int) -> bool:
        return local_mask_mod(q, k, window=real_window)

    rows = classify_all_tiles(real_sequence, block_size, mask_mod)
    total_touched = 0
    for query_block, row in enumerate(rows):
        touched = len(row["dense"]) + len(row["masked"])
        total_touched += touched
        skipped = n_blocks - touched
        start, end = query_block * block_size, (query_block + 1) * block_size - 1
        print(
            f"  query tokens {start:>4}-{end:<4}: dense (mask-free) tiles {row['dense']}, "
            f"masked tiles {row['masked']} -- {skipped} of {n_blocks} tiles skipped, zero compute"
        )

    total_tiles = n_blocks * n_blocks
    skipped_fraction = 1 - total_touched / total_tiles
    print(
        f"\nTotal: {total_touched}/{total_tiles} tiles ever touched "
        f"({skipped_fraction:.3%} skipped) -- matches lab 10's real "
        "BlockMask.sparsity() of 53.125% exactly, with zero PyTorch involved."
    )


if __name__ == "__main__":
    main()
