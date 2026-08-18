"""Path-specific ``torch.compile`` helpers with explicit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

CompilePath = Literal["training", "prefill", "decode"]


@dataclass(frozen=True, slots=True)
class CompileReport:
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
