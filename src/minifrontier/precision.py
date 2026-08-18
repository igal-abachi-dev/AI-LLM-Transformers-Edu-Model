"""Explicit precision selection shared by training, evaluation, and generation CLIs."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Literal

import torch

Precision = Literal["auto", "float32", "bfloat16"]


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    requested: Precision
    resolved: Literal["float32", "bfloat16"]
    device_type: str
    fallback_reason: str | None = None

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.resolved == "bfloat16" else torch.float32

    def autocast_context(self) -> AbstractContextManager[object]:
        if self.resolved == "bfloat16":
            return torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)
        return nullcontext()


def resolve_precision(requested: Precision, device: torch.device | str) -> PrecisionPolicy:
    torch_device = torch.device(device)
    if requested not in ("auto", "float32", "bfloat16"):
        raise ValueError(f"unknown precision: {requested}")
    if requested == "float32":
        return PrecisionPolicy(requested, "float32", torch_device.type)
    if torch_device.type == "cuda":
        supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if supported:
            return PrecisionPolicy(requested, "bfloat16", "cuda")
        return PrecisionPolicy(
            requested,
            "float32",
            "cuda",
            "CUDA BF16 is unavailable; using float32",
        )
    if requested == "bfloat16" and torch_device.type == "cpu":
        return PrecisionPolicy(requested, "bfloat16", "cpu")
    reason = None if requested == "auto" else f"BF16 autocast is unsupported on {torch_device.type}"
    return PrecisionPolicy(requested, "float32", torch_device.type, reason)


def cast_model_for_inference(
    model: torch.nn.Module,
    requested: Precision,
    device: torch.device | str,
) -> PrecisionPolicy:
    """Apply a stable weight dtype so cached projection dtype is unambiguous."""

    policy = resolve_precision(requested, device)
    model.to(device=torch.device(device), dtype=policy.dtype)
    model.eval()
    return policy
