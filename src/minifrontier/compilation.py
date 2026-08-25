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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

CompilePath = Literal["training", "prefill", "decode"]
CompileBackend = str | Callable[..., Any] | None


@dataclass(slots=True)
class CompileReport:
    """What was asked for versus what actually happened, for the run record."""

    path: CompilePath
    requested: bool
    wrapped: bool
    compiled: bool
    backend: str | None
    error: str | None = None


class _LazyCompileFallback(nn.Module):
    """Mark compilation successful only after execution and fall back safely."""

    def __init__(
        self,
        eager_module: nn.Module,
        compiled_module: nn.Module,
        report: CompileReport,
        *,
        fail_on_error: bool,
    ) -> None:
        super().__init__()
        self.compiled_module = compiled_module
        # The compiled module already owns the eager module as ``_orig_mod``. Keep
        # this second reference unregistered so parameter/state traversal does not
        # expose the same model under two prefixes.
        object.__setattr__(self, "_eager_module", eager_module)
        object.__setattr__(self, "_report", report)
        self.fail_on_error = fail_on_error
        self.use_eager_fallback = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.use_eager_fallback:
            return self._eager_module(*args, **kwargs)
        try:
            output = self.compiled_module(*args, **kwargs)
        except Exception as error:
            self._report.compiled = False
            self._report.error = f"{type(error).__name__}: {error}"
            if self.fail_on_error:
                raise RuntimeError(
                    f"torch.compile failed for {self._report.path}: {error}"
                ) from error
            self.use_eager_fallback = True
            return self._eager_module(*args, **kwargs)
        self._report.compiled = True
        self._report.error = None
        return output


def _backend_name(backend: CompileBackend) -> str | None:
    if backend is None or isinstance(backend, str):
        return backend
    return getattr(backend, "__name__", type(backend).__name__)


def maybe_compile(
    module: nn.Module,
    *,
    enabled: bool,
    path: CompilePath,
    backend: CompileBackend = None,
    fullgraph: bool = False,
    fail_on_error: bool = False,
) -> tuple[nn.Module, CompileReport]:
    """Compile one execution path without implying other paths also compile."""

    if not enabled:
        return module, CompileReport(path, False, False, False, _backend_name(backend))
    report = CompileReport(path, True, False, False, _backend_name(backend))
    try:
        compiled = torch.compile(module, backend=backend, fullgraph=fullgraph)
    except Exception as error:
        report.error = f"{type(error).__name__}: {error}"
        if fail_on_error:
            raise RuntimeError(f"torch.compile failed for {path}: {error}") from error
        return module, report
    report.wrapped = True
    return _LazyCompileFallback(
        module,
        compiled,
        report,
        fail_on_error=fail_on_error,
    ), report
