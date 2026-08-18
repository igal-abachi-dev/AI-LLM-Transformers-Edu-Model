"""Reproducibility helpers shared by tests, training, and benchmarks."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without pretending all kernels are deterministic."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]
