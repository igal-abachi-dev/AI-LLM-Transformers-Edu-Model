"""Inspect the educational Newton-Schulz step used to explain first-party Muon.

What this lab shows
-------------------
AdamW treats a weight matrix as a bag of unrelated numbers and gives each one its
own step size. Muon treats it as a matrix, and "orthogonalizes" the gradient
before applying it.

The intuition without the linear algebra: a raw gradient matrix is usually
lopsided. A couple of directions dominate and the rest are nearly ignored, so most
of the step goes into reinforcing what the model already learned. Muon evens the
directions out, so the update pushes on all of them comparably -- which often
reaches the same loss in noticeably fewer steps.

"Evening out the directions" has a precise meaning: make all the matrix's singular
values equal. The direct way to do that is an SVD, which is far too slow to run on
every weight matrix on every step. Newton-Schulz gets close using only matrix
multiplies, which is the one thing a GPU is superb at.

Run it with::

    uv run --extra cpu python labs/06_adamw_vs_muon.py

What to look for. The printed singular values of the orthogonalized matrix cluster
near 1.0 -- roughly 0.74 to 1.11 with the default five steps. That spread is the
honest picture: Newton-Schulz approximates the property rather than achieving it
exactly, and five iterations is the usual compromise between accuracy and cost.
Compare that clustering against the raw input's Frobenius norm to see how much of
the original lopsidedness was flattened out.

Note the final line. Real training calls ``torch.optim.Muon``; this readable FP32
function exists to be understood, and is never used to train anything. For an
actual AdamW-versus-Muon comparison, use ``scripts/compare_optimizers.py``, which
sweeps the learning rate for both -- the two optimizers have unrelated natural
scales, so reusing one rate would rig the result.
"""

import torch

from minifrontier.muon import newton_schulz_reference


def main() -> None:
    torch.manual_seed(55)
    # Stand-in for a gradient on some weight matrix. Muon only ever handles 2-D
    # parameters like this; embeddings and RMSNorm scales stay on AdamW.
    gradient = torch.randn(4, 6)
    orthogonalized = newton_schulz_reference(gradient)
    print("input shape:", tuple(gradient.shape))
    print("input Frobenius norm:", gradient.norm().item())
    print("Muon reference singular values:", torch.linalg.svdvals(orthogonalized).tolist())
    print("Production training uses torch.optim.Muon, never this teaching function.")


if __name__ == "__main__":
    main()
