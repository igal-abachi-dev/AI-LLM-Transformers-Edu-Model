"""Comparable experiment records and bounded throughput measurement (MF-037)."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from minifrontier.model import MiniFrontier

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ComparisonKey:
    data_id: str
    tokenizer_hash: str
    token_budget: int
    batch_tokens: int
    context_length: int
    seed_policy: str
    evaluation_id: str

    def __post_init__(self) -> None:
        for name in ("token_budget", "batch_tokens", "context_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("data_id", "tokenizer_hash", "seed_policy", "evaluation_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    run_id: str
    comparison: ComparisonKey
    quality: dict[str, float]
    training_tokens_per_second: float | None
    inference_tokens_per_second: float | None
    wall_time_seconds: float
    peak_vram_bytes: int
    kv_cache_bytes: int
    hardware: str
    notes: list[str]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        if self.wall_time_seconds < 0 or self.peak_vram_bytes < 0 or self.kv_cache_bytes < 0:
            raise ValueError("time and byte metrics cannot be negative")
        throughputs = (self.training_tokens_per_second, self.inference_tokens_per_second)
        if any(
            value is not None and (value <= 0 or not math.isfinite(value)) for value in throughputs
        ):
            raise ValueError("throughput metrics must be finite and positive when present")
        if any(not math.isfinite(value) for value in self.quality.values()):
            raise ValueError("quality metrics must be finite")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported benchmark schema version: {self.schema_version}")

    def comparable_to(self, other: BenchmarkRecord) -> bool:
        return self.schema_version == other.schema_version and self.comparison == other.comparison

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hardware_description() -> str:
    if torch.cuda.is_available():
        return f"{platform.platform()} / {torch.cuda.get_device_name(0)}"
    return f"{platform.platform()} / CPU"


def write_record(record: BenchmarkRecord, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_record(path: str | Path) -> BenchmarkRecord:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["comparison"] = ComparisonKey(**raw["comparison"])
    return BenchmarkRecord(**raw)


@torch.inference_mode()
def measure_forward_throughput(
    model: MiniFrontier,
    tokens: torch.Tensor,
    *,
    iterations: int = 3,
) -> tuple[float, float, int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    device = tokens.device
    model.eval()
    model(tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(iterations):
        model(tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    throughput = tokens.numel() * iterations / elapsed
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return throughput, elapsed, peak
