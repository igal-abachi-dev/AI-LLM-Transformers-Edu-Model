"""Scaled dot-product attention with zero external libraries -- no PyTorch, no NumPy.

What this lab shows
--------------------
`00_attention_math.py` derives attention by hand, but still leans on PyTorch tensors
to do the actual arithmetic. This lab goes one level further down: every operation
-- matrix multiply, transpose, softmax, dropout, masking -- is written out as plain
Python lists, loops, and the standard-library ``math``/``random`` modules. Nothing
here is a tensor-library trick; it's the literal arithmetic those libraries exist to
speed up.

That also makes it a Rosetta Stone for what a real GPU kernel is actually doing
underneath the abstraction, one line of C++/CUDA source at a time:

- ``CUDABlas.cpp`` / cuBLAS ``gemm`` <-> ``matmul_2d``: three nested loops computing
  ``C[i][j] = sum_k A[i][k] * B[k][j]``. Real cuBLAS tiles this same computation into
  the GPU's fast on-chip shared memory instead of three flat loops over slow global
  memory -- see this project's own "Tiles vs. tensors" section in `introduction.md`
  for why that tiling is most of what makes a fused attention kernel fast.
- ``SoftMaxKernel.cpp`` <-> ``softmax_1d``: find the row's max, exponentiate every
  shifted value, then normalize. The max-subtraction is not an optimization, it's
  required for correctness at scale -- see the comment inside ``softmax_1d``.
- ``Dropout.cu`` <-> ``dropout_1d``: zero each element with probability ``dropout_p``,
  otherwise scale it by ``1 / (1 - dropout_p)`` so the expected value is unchanged
  whether or not dropout is active. (MiniFrontier's own real `dropout` config field
  defaults to ``0.0`` -- these models are limited by how much data they see, not by
  overfitting, so dropout would only slow learning down. This function is here to
  show the real mechanism, not because a real run turns it on.)
- ``tensor.transpose(-2, -1)`` <-> ``transpose_2d``: swap row and column indices.

Run it with::

    uv run --extra cpu python labs/00b_attention_pure_python.py

What to look for. Three tokens, two-number Query/Key/Value vectors chosen so every
score is hand-checkable, under a causal mask (token i only ever sees tokens 0..i):

- Token 0 can only see itself, so softmax over one option is trivially 100% -- its
  output is exactly Value 0, no blending possible.
- Token 1 sees itself and token 0. Its query happens to be orthogonal to Key 0
  (their dot product is exactly 0), so Key 0 scores lower than Key 1 -- but "lower"
  is not "zero": softmax still gives Key 0 a real, nonzero share. The printed
  output is a genuine ~33%/67% blend of Value 0 and Value 1, not a clean 100% to
  either one. Worth sitting with, because it's a common beginner mistake to assume
  "orthogonal query/key" means "gets excluded" -- it only lowers that key's raw
  score, exactly the same way every other lower-scoring key does. Only an explicit
  mask entry produces an exact, hard `0.0` (see `softmax_1d`'s own comment on why
  `-inf` is different from merely "a low score").
- Token 2 sees all three tokens and its output is a real three-way blend, weighted
  toward whichever token scored highest (here, token 2's own value, since a token's
  query always scores highest against its own key when Q and K start out equal).

Same three-token story `00_attention_math.py` tells with PyTorch tensors, this time
with no library standing between you and the arithmetic -- and the same "exact
numbers, not just directionally right" discipline the rest of this project applies
to every claim it makes.
"""

from __future__ import annotations

import math
import random

# Type aliases for nested lists (representing tensors)
Vector1D = list[float]
Matrix2D = list[list[float]]
Tensor4D = list[list[list[list[float]]]]
Mask2D = list[list[bool]]


# =====================================================================
# 1. Low-level building blocks (Linear Algebra & Element-wise math)
# =====================================================================


def transpose_2d(matrix: Matrix2D) -> Matrix2D:
    """Transpose a 2D matrix of shape [Rows, Cols] -> [Cols, Rows].

    Equivalent to ATen's `tensor.transpose(-2, -1)`.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


def matmul_2d(A: Matrix2D, B: Matrix2D) -> Matrix2D:
    """Multiply two 2D matrices: [M, K] x [K, N] -> [M, N].

    Equivalent to cuBLAS `gemm` / `torch.matmul`.
    Every element [i][j] is the dot product of row i of A and col j of B.
    """
    # A, B, M, K, N follow standard linear-algebra / BLAS notation for a general
    # matrix multiply (the same letters cuBLAS's own `gemm` signature and most
    # attention/LLM libraries use): A is [M, K], B is [K, N], the result is [M, N].
    # M = rows of A (and of the result), K = the shared "inner" dimension being
    # summed over (A's columns / B's rows), N = columns of B (and of the result).
    M = len(A)
    K = len(A[0])
    K_b = len(B)
    N = len(B[0])

    if K_b != K:
        raise ValueError(f"Inner dimension mismatch: A is [{M}, {K}] but B is [{K_b}, {N}]")

    # Initialize resulting matrix with zeros: [M, N]
    result = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            dot = 0.0
            for k in range(K):
                dot += A[i][k] * B[k][j]
            result[i][j] = dot
    return result


def softmax_1d(row: Vector1D) -> Vector1D:
    """Numerically stable softmax over a 1D vector.

    Equivalent to ATen `SoftMaxKernel.cpp` / `SoftMax.cpp`.

    Why subtract max(row)?
    e^x grows exponentially fast. e^1000 causes floating-point overflow (`inf`).
    Since e^(x - m) / sum(e^(x - m)) == e^x / sum(e^x), subtracting the maximum
    value m keeps all exponents <= 0, bounding e^(x - m) between [0.0, 1.0].
    """
    # Step 1: Find max value (ignoring -inf)
    max_val = max(row)
    if math.isinf(max_val) and max_val < 0:
        raise ValueError("Cannot compute softmax on a row of all -inf")

    # Step 2: Compute e^(x - max) and sum them up
    exps: list[float] = []
    for x in row:
        if x == float("-inf"):
            # Masked tokens have a score of -inf, and e^(-inf) == 0.0
            exps.append(0.0)
        else:
            exps.append(math.exp(x - max_val))

    sum_exps = sum(exps)
    if sum_exps == 0.0:
        raise ValueError("Sum of exponents is zero in softmax")

    # Step 3: Normalize so elements sum to 1.0
    return [e / sum_exps for e in exps]


def dropout_1d(row: Vector1D, dropout_p: float, training: bool) -> Vector1D:
    """Inverted dropout on a 1D vector.

    Equivalent to ATen `Dropout.cpp` / CUDA `Dropout.cu`.

    During training:
      - Each element is zeroed out with probability `dropout_p`.
      - Kept elements are scaled by `1 / (1 - dropout_p)` so that the
        expected value remains unchanged across training and inference.
    """
    if not training or dropout_p == 0.0:
        return list(row)

    keep_prob = 1.0 - dropout_p
    scale = 1.0 / keep_prob

    result: list[float] = []
    for x in row:
        # random.random() returns a uniform float in [0.0, 1.0)
        if random.random() < dropout_p:
            result.append(0.0)
        else:
            result.append(x * scale)
    return result


# =====================================================================
# 2. Shape Inspection Helpers
# =====================================================================


def get_shape_4d(t: Tensor4D) -> tuple[int, int, int, int]:
    """Return [Batch, Heads, Sequence, Head_Dim] from nested lists."""
    # b = batch size, h = number of heads, s = sequence length, d = head_dim.
    # Short names on purpose: every call site below immediately unpacks them into
    # a shape tuple that's already spelled out in this docstring and re-explained
    # at each call site (e.g. `sq`/`sk` for query/key sequence length below).
    b = len(t)
    h = len(t[0]) if b > 0 else 0
    s = len(t[0][0]) if h > 0 else 0
    d = len(t[0][0][0]) if s > 0 else 0
    return b, h, s, d


def get_shape_2d(m: Mask2D) -> tuple[int, int]:
    """Return [Rows, Cols] from a 2D list."""
    # r = rows (query_sequence for a mask), c = cols (key_sequence for a mask).
    r = len(m)
    c = len(m[0]) if r > 0 else 0
    return r, c


# =====================================================================
# 3. Main Scaled Dot-Product Attention Function
# =====================================================================


def manual_scaled_dot_product_attention_pure_python(
    query: Tensor4D,
    key: Tensor4D,
    value: Tensor4D,
    *,
    mask: Mask2D,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tensor4D:
    """Pure Python attention over [batch, heads, sequence, head_dim].

    Exact equivalent of:
    ``softmax(Q @ K^T / sqrt(head_dim) + mask) @ V``

    Named with a `_pure_python` suffix (rather than reusing
    ``attention.py``'s own ``manual_scaled_dot_product_attention``) because that name
    is the real, torch-tensor-based teaching reference this repo actually tests
    production code against -- this function is one level further removed, useful for
    reading, never imported by the real model.

    Args:
        query: [B, H, Sq, D] float list
        key:   [B, H, Sk, D] float list
        value: [B, H, Sk, D] float list
        mask:  [Sq, Sk] bool list (True = allowed, False = masked out)
        dropout_p: dropout probability in [0, 1)
        training: whether dropout is active

    Returns:
        [B, H, Sq, D] float list -- one output vector per query position.
    """
    # -------------------------------------------------------------
    # 1. Validation checks
    # -------------------------------------------------------------
    # B/H = batch size / head count, Sq/Sk = query/key sequence length, D = head_dim.
    b_q, h_q, sq, d_q = get_shape_4d(query)
    b_k, h_k, sk, d_k = get_shape_4d(key)
    b_v, h_v, sv, d_v = get_shape_4d(value)

    if (b_k, h_k, sk, d_k) != (b_v, h_v, sv, d_v):
        raise ValueError("key and value shapes must match")
    if b_q != b_k or d_q != d_k:
        raise ValueError("query and key batch/head dimensions are incompatible")
    if h_q != h_k:
        raise ValueError("manual attention requires equal expanded query and KV head counts")

    mask_r, mask_c = get_shape_2d(mask)
    if (mask_r, mask_c) != (sq, sk):
        raise ValueError(
            f"mask must have shape [query_sequence, key_sequence], got [{mask_r}, {mask_c}]"
        )
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")

    # Ensure every query attends to at least one key
    for q_idx, row in enumerate(mask):
        if not any(row):
            raise ValueError(f"query token {q_idx} has no visible keys (all masked out)")

    # -------------------------------------------------------------
    # 2. Attention computation
    # -------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_q)

    # Output shape: [B, H, Sq, D]
    output: Tensor4D = []

    for b in range(b_q):
        batch_out: list[Matrix2D] = []
        for h in range(h_q):
            # Extract 2D slices for this head:
            # Q_head: [Sq, D], K_head: [Sk, D], V_head: [Sk, D]
            Q_head = query[b][h]
            K_head = key[b][h]
            V_head = value[b][h]

            # Step A: K^T (transpose key head): [Sk, D] -> [D, Sk]
            K_T = transpose_2d(K_head)

            # Step B: MatMul Q @ K^T -> [Sq, Sk]
            # Each entry (i, j) is the dot product of Query[i] and Key[j]
            raw_scores = matmul_2d(Q_head, K_T)

            # Step C: Scale and Apply Mask
            # Disallowed pairs become -inf, turning into 0% probability under softmax
            masked_scores: Matrix2D = []
            for i in range(sq):
                score_row: list[float] = []
                for j in range(sk):
                    scaled_val = raw_scores[i][j] * scale
                    if not mask[i][j]:
                        score_row.append(float("-inf"))
                    else:
                        score_row.append(scaled_val)
                masked_scores.append(score_row)

            # Step D: Softmax & Dropout along the last dimension (key tokens)
            # Each query's row turns into a probability distribution summing to 1.0
            probabilities: Matrix2D = []
            for i in range(sq):
                probs = softmax_1d(masked_scores[i])
                probs = dropout_1d(probs, dropout_p=dropout_p, training=training)
                probabilities.append(probs)

            # Step E: MatMul Probabilities @ V -> [Sq, D]
            # This is the weighted average of the value vectors!
            head_output = matmul_2d(probabilities, V_head)
            batch_out.append(head_output)

        output.append(batch_out)

    return output


def main() -> None:
    # Sequence=3 tokens, Head_Dim=2. Q and K share the same three vectors on
    # purpose -- it makes "a token's query scores highest against its own key"
    # (the diagonal of Q @ K^T) easy to verify by eye in the printed matrix below.
    Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 20.0], [30.0, 30.0]]

    # Causal mask (lower triangular): token i may only look at tokens 0..i.
    causal_mask = [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]

    print("Query:", Q)
    print("Key:  ", K)
    print("Value:", V)
    print("Causal mask (True = allowed):", causal_mask)

    # Step A: K^T -- [3, 2] -> [2, 3].
    K_T = transpose_2d(K)

    # Step B: raw_scores[i][j] = dot(Query[i], Key[j]) -- [3, 3].
    raw_scores = matmul_2d(Q, K_T)
    print("\nRaw Q @ K^T:")
    for row in raw_scores:
        print(" ", [round(v, 4) for v in row])

    # Step C: scale by 1/sqrt(head_dim), then send every masked-out (future)
    # pair to -inf so it becomes an exact, hard 0.0 once softmax runs -- not
    # merely a small number, which is why -inf is used instead of e.g. -1e9.
    scale = 1.0 / math.sqrt(len(Q[0]))
    masked_scores: Matrix2D = []
    for i, row in enumerate(raw_scores):
        masked_scores.append(
            [row[j] * scale if causal_mask[i][j] else float("-inf") for j in range(len(row))]
        )
    print(f"\nScaled (x{scale:.4f}) and masked scores:")
    for row in masked_scores:
        print(" ", [round(v, 4) if v != float("-inf") else "-inf" for v in row])

    # Step D: softmax turns each row into a probability distribution over keys.
    probabilities = [softmax_1d(row) for row in masked_scores]
    print("\nAttention probabilities (each row sums to 1.0):")
    for row in probabilities:
        print(" ", [round(v, 4) for v in row])

    # Step E: each token's output is that row's weighted blend of Value vectors.
    output = matmul_2d(probabilities, V)
    print("\nOutput = probabilities @ V:")
    for token_idx, row in enumerate(output):
        print(f"  Token {token_idx}: {[round(v, 4) for v in row]}")

    # `matmul_2d(probabilities, V)` above is not a separate idea from "a weighted
    # blend of Value vectors" -- it IS that, just computed a whole row at a time.
    # Spelled out one token at a time, for token 1 specifically:
    p0, p1 = probabilities[1][0], probabilities[1][1]
    manual_blend = [p0 * V[0][d] + p1 * V[1][d] for d in range(len(V[0]))]
    print(
        f"\nToken 1's output, the long way: {p0:.4f} * {V[0]} + {p1:.4f} * {V[1]}"
        f" = {[round(v, 4) for v in manual_blend]}"
    )
    assert all(abs(a - b) < 1e-9 for a, b in zip(manual_blend, output[1], strict=True))

    # Token 0 can only see itself, so softmax over exactly one option is
    # trivially [1.0, 0.0, 0.0] -- this is exact, not approximate, regardless of
    # what the actual Query/Key numbers are, so it's safe to assert on the nose.
    assert probabilities[0] == [1.0, 0.0, 0.0]
    assert output[0] == [10.0, 0.0]

    # Cross-check this step-by-step derivation against the batched,
    # four-dimensional entry point -- the shape real (production) code passes
    # around -- to prove the two agree beyond just this one worked example.
    batched_output = manual_scaled_dot_product_attention_pure_python(
        [[Q]], [[K]], [[V]], mask=causal_mask
    )
    for i in range(len(output)):
        for d in range(len(output[i])):
            assert abs(batched_output[0][0][i][d] - output[i][d]) < 1e-9
    print(
        "\nConfirmed: the step-by-step derivation above matches "
        "manual_scaled_dot_product_attention_pure_python() exactly."
    )


if __name__ == "__main__":
    main()
