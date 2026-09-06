"""Reproducibility helpers shared by tests, training, and benchmarks.

Beginner's map of this file
---------------------------
Neural networks are full of randomness: the initial weights, the shuffling of
data, sampling during generation. All of it comes from pseudo-random generators,
which produce a fixed sequence once you fix their starting **seed**.

Seeding everything means two runs of the same code produce the same numbers. That
matters for two reasons here: a test that fails only sometimes is nearly
impossible to debug, and an experiment comparing A against B is worthless if the
difference could just be luck.

Note that determinism is not free -- some fast GPU kernels add results in a
nondeterministic order -- so this is a deliberate choice made per run rather than
a global default.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without pretending all kernels are deterministic.

    ``torch.use_deterministic_algorithms`` is a global, process-wide flag with no
    scoped/context-manager equivalent (unlike ``autocast``), so it is always set
    explicitly here to exactly ``deterministic`` -- never only turned on and left
    that way. Every real script in this project currently calls this once with one
    consistent value for its whole run, so this has not caused a real failure, but
    a future script mixing a CPU-deterministic phase with a later CUDA phase in the
    same process would otherwise inherit a stuck ``True`` from the first phase --
    and PyTorch does not silently degrade to nondeterministic in that state, it
    raises on any op with no deterministic CUDA implementation.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        # CUBLAS_WORKSPACE_CONFIG only takes effect if set before cuBLAS
        # initializes, so there is no matching "unset" action once a CUDA
        # context already exists -- setdefault (not overwrite) is deliberate,
        # to respect a caller who configured this themselves beforehand.
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
