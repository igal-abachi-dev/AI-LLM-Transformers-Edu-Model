"""Path-specific ``torch.compile`` helpers with explicit diagnostics.

Beginner's map of this file
---------------------------
``torch.compile`` traces a model's Python into a graph and generates fused
kernels for it, which can be substantially faster. It is optional here on
purpose: eager execution stays the correctness baseline, and anything compiled
has to prove it produces the same numbers.

Why "path-specific"? A model has three quite different execution shapes --
training, prefilling a prompt, and decoding one token at a time -- and compiling
one says nothing about the others. Each gets its own attempt and its own report,
so a run record cannot claim more than was actually measured.

Compilation is also allowed to fail. By default a failure is captured in the
report and the original eager module is returned, because a slower run that works
beats a fast one that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

CompilePath = Literal["training", "prefill", "decode"]


@dataclass(frozen=True, slots=True)
class CompileReport:
    """What was asked for versus what actually happened, for the run record."""

    path: CompilePath
    requested: bool
    compiled: bool
    backend: str | None
    error: str | None = None


def maybe_compile(
    module: nn.Module,
    *,
    enabled: bool,
    path: CompilePath,
    backend: str | None = None,
    fullgraph: bool = False,
    fail_on_error: bool = False,
) -> tuple[nn.Module, CompileReport]:
    """Compile one execution path without implying other paths also compile."""

    if not enabled:
        return module, CompileReport(path, False, False, backend)
    try:
        compiled = torch.compile(module, backend=backend, fullgraph=fullgraph)
    except Exception as error:
        if fail_on_error:
            raise RuntimeError(f"torch.compile failed for {path}: {error}") from error
        return module, CompileReport(path, True, False, backend, f"{type(error).__name__}: {error}")
    return compiled, CompileReport(path, True, True, backend)
