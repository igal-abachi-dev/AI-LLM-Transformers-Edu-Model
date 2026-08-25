"""Explicit precision selection shared by training, evaluation, and generation CLIs.

Beginner's map of this file
---------------------------
How many bits each number gets.

* **FP32** (32 bits) -- the safe default and this project's correctness baseline.
* **BF16** (16 bits) -- half the memory and much faster on GPUs with native BF16
  Tensor Cores (Ampere and newer). It keeps FP32's *range* (so values do not
  overflow to infinity) and gives up *precision* instead, which turns out to be
  the right trade for neural networks.
* **FP16** (16 bits) -- half the memory, and the precision Turing-class Tensor
  Cores (GTX 16-series, RTX 20-series) actually accelerate; those cards have no
  native BF16 matrix-multiply hardware and can only *emulate* it in software. FP16's
  narrower exponent range means small gradients can underflow to zero, which is
  why FP16 training needs a loss scaler (see ``training.py``) while BF16 does not.

Training in BF16/FP16 does not mean everything is that dtype. ``torch.autocast``
runs the matmuls in the lower precision while keeping the parts that need
accuracy -- reductions, the loss, the optimizer's own state -- in FP32. That is
the "mixed" in mixed precision.

The reason this is a whole module rather than one flag: reduced precision needs
matching hardware, and quietly falling back -- or quietly running emulated --
would make a benchmark meaningless. Every policy here records what was
*requested*, what was *resolved*, and why they differ, so a run record can state
it plainly.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Literal

import torch

Precision = Literal["auto", "float32", "bfloat16", "float16"]
_ResolvedPrecision = Literal["float32", "bfloat16", "float16"]

_DTYPE_BY_RESOLVED: dict[_ResolvedPrecision, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    requested: Precision
    resolved: _ResolvedPrecision
    device_type: str
    fallback_reason: str | None = None

    @property
    def dtype(self) -> torch.dtype:
        return _DTYPE_BY_RESOLVED[self.resolved]

    @property
    def needs_grad_scaler(self) -> bool:
        """FP16's narrow exponent range can underflow small gradients to zero;

        BF16 keeps FP32's exponent range and never needs this."""

        return self.resolved == "float16"

    def autocast_context(self) -> AbstractContextManager[object]:
        """Context manager for the forward pass; a no-op when running in FP32."""

        if self.resolved in ("bfloat16", "float16"):
            return torch.autocast(device_type=self.device_type, dtype=self.dtype)
        return nullcontext()


def resolve_precision(requested: Precision, device: torch.device | str) -> PrecisionPolicy:
    """Decide the real precision for this device and record why.

    ``"auto"`` prefers genuine Tensor Core acceleration over PyTorch's
    emulation-inclusive BF16 check: native BF16 (Ampere+) first, then FP16 on any
    other CUDA device, then FP32 everywhere else -- which is what makes the same
    command work well on a CPU dev box, a Turing card, and an Ampere+ machine.
    Explicit requests are always honored as asked; only ``"auto"`` chooses between
    BF16 and FP16 on CUDA. An explicit ``"bfloat16"`` request on a non-native card
    still resolves to emulated BF16 rather than being silently redirected to
    FP16 -- least surprising for a caller who asked for BF16 specifically.
    """

    torch_device = torch.device(device)
    if requested not in ("auto", "float32", "bfloat16", "float16"):
        raise ValueError(f"unknown precision: {requested}")
    if requested == "float32":
        return PrecisionPolicy(requested, "float32", torch_device.type)

    if torch_device.type == "cuda":
        cuda_ready = torch.cuda.is_available()
        if requested == "bfloat16":
            if cuda_ready and torch.cuda.is_bf16_supported():
                return PrecisionPolicy(requested, "bfloat16", "cuda")
            return PrecisionPolicy(
                requested, "float32", "cuda", "CUDA BF16 is unavailable; using float32"
            )
        if requested == "float16":
            if cuda_ready:
                return PrecisionPolicy(requested, "float16", "cuda")
            return PrecisionPolicy(
                requested, "float32", "cuda", "CUDA FP16 is unavailable; using float32"
            )
        # "auto": prefer hardware that actually accelerates the dtype, not PyTorch's
        # emulation-inclusive is_bf16_supported() default, which reports True on
        # Turing even though Turing has no BF16 matrix-multiply Tensor Cores.
        if cuda_ready and torch.cuda.is_bf16_supported(including_emulation=False):
            return PrecisionPolicy(requested, "bfloat16", "cuda")
        if cuda_ready:
            return PrecisionPolicy(requested, "float16", "cuda")
        return PrecisionPolicy(requested, "float32", "cuda", "no CUDA device is available")

    if requested == "bfloat16" and torch_device.type == "cpu":
        return PrecisionPolicy(requested, "bfloat16", "cpu")
    if requested == "float16":
        # PyTorch's CPU FP16 op coverage is far weaker/newer than its CPU BF16
        # coverage, so unlike BF16, FP16 does not get a CPU path here.
        reason = f"FP16 autocast is not reliably supported on {torch_device.type}"
        return PrecisionPolicy(requested, "float32", torch_device.type, reason)
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
