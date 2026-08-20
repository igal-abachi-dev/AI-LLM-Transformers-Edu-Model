"""Explicit precision selection shared by training, evaluation, and generation CLIs.

Beginner's map of this file
---------------------------
How many bits each number gets.

* **FP32** (32 bits) -- the safe default and this project's correctness baseline.
* **BF16** (16 bits) -- half the memory and much faster on modern GPUs. It keeps
  FP32's *range* (so values do not overflow to infinity) and gives up *precision*
  instead, which turns out to be the right trade for neural networks.

Training in BF16 does not mean everything is BF16. ``torch.autocast`` runs the
matmuls in BF16 while keeping the parts that need accuracy -- reductions, the
loss, the optimizer's own state -- in FP32. That is the "mixed" in mixed
precision.

The reason this is a whole module rather than one flag: BF16 needs hardware
support, and quietly falling back would make a benchmark meaningless. Every
policy here records what was *requested*, what was *resolved*, and why they
differ, so a run record can state it plainly.
"""

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
        """Context manager for the forward pass; a no-op when running in FP32."""

        if self.resolved == "bfloat16":
            return torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)
        return nullcontext()


def resolve_precision(requested: Precision, device: torch.device | str) -> PrecisionPolicy:
    """Decide the real precision for this device and record why.

    ``"auto"`` means BF16 on a GPU that supports it and FP32 everywhere else --
    which is what makes the same command work on a CPU dev box and an RTX machine.
    """

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
    """Apply a stable weight dtype so cached projection dtype is unambiguous.

    Inference casts the weights themselves rather than using autocast. That keeps
    the dtype of everything the KV cache stores predictable: the cache adopts the
    dtype the projections produce, and a cache that disagrees with the model is a
    confusing mid-generation failure.
    """

    policy = resolve_precision(requested, device)
    model.to(device=torch.device(device), dtype=policy.dtype)
    model.eval()
    return policy
