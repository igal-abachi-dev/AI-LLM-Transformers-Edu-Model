"""Schema-checked, JSON-safe metadata for reproducible runs.

Beginner's map of this file
---------------------------
Every training run, benchmark and evaluation writes one of these JSON records:
what was run, on what hardware, with which seed, for how many tokens, and what
came out. It is the difference between "the Modern model seemed faster" and a
result somebody else can check.

Note also what is deliberately *not* recorded: no environment variables, no
tokens, no local paths. A run record is meant to be publishable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


@dataclass(slots=True)
class RunMetadata:
    name: str
    config: dict[str, Any]
    seed: int
    parameters: int
    train_tokens: int = 0
    wall_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    tokens_per_second: float = 0.0
    train_loss: float | None = None
    val_loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str | None = field(default_factory=_git_revision)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    torch_version: str = field(default_factory=lambda: torch.__version__)
    platform: str = field(default_factory=platform.platform)
    device: str = field(
        default_factory=lambda: (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("run name cannot be empty")
        if self.seed < 0 or self.parameters < 0 or self.train_tokens < 0:
            raise ValueError("seed, parameters, and train_tokens must be non-negative")
        if self.wall_seconds < 0 or self.peak_memory_mb < 0 or self.tokens_per_second < 0:
            raise ValueError("timing, memory, and throughput must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        target.write_text(serialized, encoding="utf-8")
