"""Derive manual attention by hand on tensors small enough to trace with a pencil.

What this lab shows
--------------------
Attention is four steps, and this script prints every intermediate tensor so none
of them stay hidden inside a matmul:

1. ``scores = Q @ K^T`` -- one raw similarity number per (query, key) pair.
2. ``scores = scores / sqrt(head_dim)`` -- shrink before softmax, or a large dot
   product saturates softmax into "100% one token, 0% the rest" and gradients die.
3. ``scores = scores.masked_fill(~mask, -inf)`` -- future keys become impossible,
   not merely discouraged: ``-inf`` softmaxes to exactly ``0.0``.
4. ``probabilities = softmax(scores)``; ``output = probabilities @ V`` -- each
   query's row of probabilities sums to 1, and the output is that weighted blend
   of Value vectors.

Run it with::

    uv run --extra cpu python labs/00_attention_math.py

What to look for. Three tokens, two-number Query/Key/Value vectors chosen so the
scores come out as small round numbers. The script computes the by-hand version
with plain tensor ops, then checks it against ``manual_scaled_dot_product_attention``
(the readable production reference in ``attention.py``) and against PyTorch's fused
``F.scaled_dot_product_attention``. All three must agree, and the printed
probability grid should show a hard zero everywhere the causal mask forbids a
future key -- token 0's row attends only to itself, token 1's row is split between
tokens 0 and 1, and only token 2 ever sees all three.
"""

import math

import torch
from torch.nn import functional as F

from minifrontier.attention import manual_scaled_dot_product_attention
from minifrontier.masking import build_attention_mask


def main() -> None:
    # [batch=1, heads=1, sequence=3, head_dim=2]. Small integers so every step's
    # numbers stay easy to re-check with a calculator.
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]]])
    value = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]])
    mask = build_attention_mask(3, 3)  # the causal lower-triangle: query i sees keys 0..i

    print("Query:\n", query)
    print("Key:\n", key)
    print("Value:\n", value)
    print("Causal mask (True = allowed):\n", mask)

    # Step 1+2: raw similarity, then scaled by 1/sqrt(head_dim) so it does not grow
    # with head_dim before softmax ever sees it.
    scale = 1.0 / math.sqrt(query.shape[-1])
    raw_scores = torch.matmul(query, key.transpose(-2, -1))
    scaled_scores = raw_scores * scale
    print(f"\nRaw Q @ K^T (scale=1/sqrt(2)={scale:.4f}):\n", raw_scores[0, 0])
    print("Scaled scores:\n", scaled_scores[0, 0])

    # Step 3: forbidden pairs become -inf so softmax turns them into a hard zero,
    # not merely a small number.
    masked_scores = scaled_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    print("Masked scores (-inf where the mask forbids a peek):\n", masked_scores[0, 0])

    # Step 4: softmax turns each row of scores into a probability distribution,
    # then that distribution picks out a weighted blend of Value rows.
    probabilities = F.softmax(masked_scores, dim=-1)
    output = torch.matmul(probabilities, value)
    print("Attention probabilities (each row sums to 1):\n", probabilities[0, 0])
    print("Output = probabilities @ V:\n", output[0, 0])

    zero_mask = ~mask
    if zero_mask.any():
        assert torch.equal(
            probabilities[0, 0][zero_mask], torch.zeros_like(probabilities[0, 0][zero_mask])
        )
        print("\nConfirmed: every masked-out (query, key) pair got exactly 0.0 probability.")

    # Cross-check the by-hand derivation above against the two production paths.
    reference = manual_scaled_dot_product_attention(query, key, value, mask=mask)
    fused = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask.unsqueeze(0).unsqueeze(0)
    )
    assert torch.allclose(output, reference, atol=1e-6)
    assert torch.allclose(output, fused, atol=1e-6)
    print(
        "\nBy-hand output matches manual_scaled_dot_product_attention and "
        "F.scaled_dot_product_attention exactly."
    )


if __name__ == "__main__":
    main()
