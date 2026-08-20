"""Educational Muon math and first-party production optimizer partitioning.

Beginner's map of this file
---------------------------
Muon is a newer optimizer that treats a weight *matrix* as a matrix rather than
as a bag of unrelated numbers. AdamW scales each weight individually; Muon takes
the whole gradient matrix and "orthogonalizes" it before applying it.

The intuition, without the linear algebra: a raw gradient matrix is usually
lopsided -- a couple of directions dominate and the rest are nearly ignored, so
most of the step goes into repeating what the model already learned. Muon evens
the directions out so the update pushes on all of them comparably. In practice
that often reaches the same loss in noticeably fewer steps.

Two things it is not used for, and why the ``partition`` function below exists:

* **Embeddings** -- rows are independent token identities, not a transformation,
  so "even out the directions" is meaningless there.
* **1-D parameters** (RMSNorm scales) -- a vector has no directions to balance.

Those stay on AdamW, so a Muon run is really two optimizers side by side over a
provably disjoint split of the weights. In this repository Muon is an
*experiment*: AdamW remains the baseline that every claim is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from minifrontier.model import MiniFrontier
from minifrontier.training import CombinedOptimizer, TrainingConfig


def newton_schulz_reference(
    matrix: torch.Tensor,
    *,
    steps: int = 5,
    coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return the annotated Muon Newton-Schulz orthogonalized 2-D update.

    Newton-Schulz is an iteration that pushes a matrix toward having all its
    singular values equal to 1 -- that is what "orthogonalize" means here -- using
    only matrix multiplies. That matters because the direct route (an SVD) is far
    too slow to run on every weight matrix of every step, while matmuls are the one
    thing a GPU is superb at. Five iterations get close enough.

    ``labs/06_adamw_vs_muon.py`` prints the singular values so you can watch them
    converge toward 1.

    This readable FP32 function is lab-only. Production construction below calls
    ``torch.optim.Muon`` directly and never selects this implementation.
    """

    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz reference requires a 2-D matrix")
    if steps <= 0 or eps <= 0:
        raise ValueError("steps and eps must be positive")
    # The iteration is cheaper when the matrix is wide, so flip a tall one and
    # flip the answer back at the end. The result is mathematically the same.
    transposed = matrix.shape[0] > matrix.shape[1]
    update = matrix.float().mT if transposed else matrix.float()
    # Scale into the range where the iteration converges; clamp_min guards a
    # zero gradient from producing a division by zero.
    update = update / update.norm().clamp_min(eps)
    a, b, c = coefficients
    for _ in range(steps):
        # A quintic polynomial in the matrix, applied via `gram = X @ X^T`. The
        # coefficients are tuned so that repeating this drives every singular value
        # toward 1 quickly, without ever needing an SVD.
        gram = update @ update.mT
        update = a * update + (b * gram + c * (gram @ gram)) @ update
    result = update.mT if transposed else update
    return result.to(matrix.dtype)


@dataclass(frozen=True, slots=True)
class MuonPartition:
    muon_parameters: tuple[torch.nn.Parameter, ...]
    adamw_parameters: tuple[torch.nn.Parameter, ...]
    muon_names: tuple[str, ...]
    adamw_names: tuple[str, ...]


def partition_muon_parameters(model: MiniFrontier) -> MuonPartition:
    """Put hidden 2-D projection matrices in Muon and everything else in AdamW.

    The check at the end is the point of the function: every trainable parameter
    must appear exactly once across the two lists. If a weight landed in both, two
    optimizers would fight over it each step; if it landed in neither, it would
    silently never train at all. Both failures are nearly invisible in the loss
    curve, so they are made loud here instead.
    """

    muon_parameters: list[torch.nn.Parameter] = []
    adamw_parameters: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    adamw_names: list[str] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        eligible = parameter.ndim == 2 and name != "token_embedding.weight"
        if eligible:
            muon_parameters.append(parameter)
            muon_names.append(name)
        else:
            adamw_parameters.append(parameter)
            adamw_names.append(name)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = {id(parameter) for parameter in (*muon_parameters, *adamw_parameters)}
    if actual != expected or len(actual) != len(muon_parameters) + len(adamw_parameters):
        raise RuntimeError(
            "Muon/AdamW partition must contain every trainable parameter exactly once"
        )
    return MuonPartition(
        tuple(muon_parameters),
        tuple(adamw_parameters),
        tuple(muon_names),
        tuple(adamw_names),
    )


def build_muon_adamw(
    model: MiniFrontier,
    config: TrainingConfig,
    *,
    muon_learning_rate: float,
    adamw_learning_rate: float,
    momentum: float = 0.95,
    match_rms_adamw: bool = False,
) -> tuple[CombinedOptimizer, dict[str, Any]]:
    """Build first-party Muon plus AdamW over a proven-disjoint partition.

    The two optimizers need genuinely different learning rates -- Muon's update is
    normalized, so its natural scale is unrelated to AdamW's -- which is why an
    honest AdamW-versus-Muon comparison has to sweep both rather than reusing one
    number. ``scripts/compare_optimizers.py`` does exactly that.

    ``lr_scale`` is how one shared warmup/cosine schedule drives both groups at
    their own rates; the training loop multiplies by it in ``training.py``.
    """

    if muon_learning_rate <= 0 or adamw_learning_rate <= 0:
        raise ValueError("optimizer learning rates must be positive")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("the pinned PyTorch build does not provide torch.optim.Muon")
    partition = partition_muon_parameters(model)
    muon = torch.optim.Muon(
        partition.muon_parameters,
        lr=muon_learning_rate,
        weight_decay=config.weight_decay,
        momentum=momentum,
        adjust_lr_fn="match_rms_adamw" if match_rms_adamw else None,
    )
    adamw_decay = []
    adamw_no_decay = []
    for name, parameter in zip(
        partition.adamw_names,
        partition.adamw_parameters,
        strict=True,
    ):
        if name == "token_embedding.weight" and config.decay_embeddings:
            adamw_decay.append(parameter)
        else:
            adamw_no_decay.append(parameter)
    adamw = torch.optim.AdamW(
        [
            {"params": adamw_decay, "weight_decay": config.weight_decay},
            {"params": adamw_no_decay, "weight_decay": 0.0},
        ],
        lr=adamw_learning_rate,
        betas=(config.beta1, config.beta2),
    )
    peak = config.learning_rate
    for group in muon.param_groups:
        group["lr_scale"] = muon_learning_rate / peak
    for group in adamw.param_groups:
        group["lr_scale"] = adamw_learning_rate / peak
    report = {
        "implementation": "torch.optim.Muon",
        "match_rms_adamw": match_rms_adamw,
        "muon_learning_rate": muon_learning_rate,
        "adamw_learning_rate": adamw_learning_rate,
        "muon_names": list(partition.muon_names),
        "adamw_names": list(partition.adamw_names),
        "adamw_decay_names": ["token_embedding.weight"] if adamw_decay else [],
        "muon_parameters": sum(parameter.numel() for parameter in partition.muon_parameters),
        "adamw_parameters": sum(parameter.numel() for parameter in partition.adamw_parameters),
    }
    return CombinedOptimizer(muon, adamw), report
