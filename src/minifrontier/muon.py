"""Educational Muon math and first-party production optimizer partitioning."""

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

    This readable FP32 function is lab-only. Production construction below calls
    ``torch.optim.Muon`` directly and never selects this implementation.
    """

    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz reference requires a 2-D matrix")
    if steps <= 0 or eps <= 0:
        raise ValueError("steps and eps must be positive")
    transposed = matrix.shape[0] > matrix.shape[1]
    update = matrix.float().mT if transposed else matrix.float()
    update = update / update.norm().clamp_min(eps)
    a, b, c = coefficients
    for _ in range(steps):
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
    """Put hidden 2-D projection matrices in Muon and everything else in AdamW."""

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
    """Build first-party Muon plus AdamW over a proven-disjoint partition."""

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
