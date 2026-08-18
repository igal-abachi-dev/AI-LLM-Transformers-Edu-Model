"""Fail-closed accounting and decision records for optional post-V1 scale checks."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from minifrontier.checkpoint import load_training_checkpoint, save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier

Measurement = float | int | Literal["unmeasured"]


@dataclass(frozen=True, slots=True)
class ScaleStopCriteria:
    """Predeclared conditions that stop a bounded scale run without hiding the failure."""

    max_peak_vram_fraction: float = 0.95
    max_wall_hours: float = 24.0
    max_oom_failures: int = 1
    max_nonfinite_updates: int = 0
    min_training_updates: int = 20

    def __post_init__(self) -> None:
        if not 0 < self.max_peak_vram_fraction <= 1:
            raise ValueError("max_peak_vram_fraction must be in (0, 1]")
        if self.max_wall_hours <= 0 or self.max_oom_failures < 0:
            raise ValueError("wall time must be positive and OOM failures non-negative")
        if self.max_nonfinite_updates < 0 or self.min_training_updates <= 0:
            raise ValueError("invalid non-finite/update stop criteria")


@dataclass(frozen=True, slots=True)
class ScaleEstimates:
    """Analytic byte counts; activations and allocator overhead are intentionally excluded."""

    parameter_count: int
    parameter_bytes: int
    gradient_bytes: int
    adamw_state_bytes: int
    training_lower_bound_bytes: int
    full_history_kv_bytes: int
    bounded_local_kv_bytes: int
    inference_full_history_lower_bound_bytes: int
    inference_bounded_local_lower_bound_bytes: int
    assumptions: tuple[str, ...]


def exact_parameter_count(config: ModelConfig) -> int:
    """Construct on the meta device so exact accounting never allocates scale-sized storage."""

    with torch.device("meta"):
        model = MiniFrontier(config)
    return model.parameter_count()


def kv_cache_bytes(
    config: ModelConfig,
    *,
    batch_size: int,
    context_length: int,
    bytes_per_element: int,
    bounded_local: bool,
) -> int:
    if batch_size <= 0 or not 0 < context_length <= config.max_seq_len:
        raise ValueError("invalid batch size or context length for KV accounting")
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")
    capacity_sum = 0
    for layer_index in range(config.n_layers):
        if bounded_local and config.is_local_layer(layer_index):
            capacity_sum += min(context_length, config.local_window)
        else:
            capacity_sum += context_length
    elements = 2 * batch_size * config.n_kv_heads * config.head_dim * capacity_sum
    return elements * bytes_per_element


def estimate_scale(
    config: ModelConfig,
    *,
    batch_size: int = 1,
    context_length: int | None = None,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    adamw_state_bytes: int = 8,
) -> ScaleEstimates:
    context = context_length or config.max_seq_len
    count = exact_parameter_count(config)
    weights = count * parameter_bytes
    gradients = count * gradient_bytes
    optimizer = count * adamw_state_bytes
    full_kv = kv_cache_bytes(
        config,
        batch_size=batch_size,
        context_length=context,
        bytes_per_element=parameter_bytes,
        bounded_local=False,
    )
    bounded_kv = kv_cache_bytes(
        config,
        batch_size=batch_size,
        context_length=context,
        bytes_per_element=parameter_bytes,
        bounded_local=True,
    )
    return ScaleEstimates(
        parameter_count=count,
        parameter_bytes=weights,
        gradient_bytes=gradients,
        adamw_state_bytes=optimizer,
        training_lower_bound_bytes=weights + gradients + optimizer,
        full_history_kv_bytes=full_kv,
        bounded_local_kv_bytes=bounded_kv,
        inference_full_history_lower_bound_bytes=weights + full_kv,
        inference_bounded_local_lower_bound_bytes=weights + bounded_kv,
        assumptions=(
            "analytic lower bounds, not measured allocator or peak VRAM",
            "BF16 parameters/gradients and FP32 AdamW first/second moments by default",
            "training estimate excludes activations, temporary kernels, CUDA context, "
            "and fragmentation",
            "KV estimate includes K and V only and uses compact n_kv_heads",
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_checkpoint_smoke() -> dict[str, object]:
    """Exercise forward/backward plus trusted-local checkpoint wiring on a tiny projection."""

    torch.manual_seed(7)
    config = ModelConfig.tiny_modern(attention_impl="sdpa", max_seq_len=12, local_window=4)
    model = MiniFrontier(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    loss = model(tokens, labels=tokens).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    with tempfile.TemporaryDirectory(prefix="minifrontier-scale-") as directory:
        checkpoint = Path(directory) / "checkpoint"
        save_training_checkpoint(
            checkpoint,
            model,
            optimizer=optimizer,
            trainer_state={"completed_updates": 1},
            data_cursor={"cursor": 1},
        )
        restored = MiniFrontier(config)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        trainer_state, cursor = load_training_checkpoint(
            checkpoint,
            restored,
            optimizer=restored_optimizer,
            restore_rng=False,
            trusted_local_state=True,
        )
        with torch.no_grad():
            parity = torch.equal(model(tokens).logits, restored(tokens).logits)
    return {
        "status": "passed" if parity else "failed",
        "projection": "tiny_modern_not_scale_performance",
        "loss_finite": bool(torch.isfinite(loss)),
        "checkpoint_logits_exact": parity,
        "trainer_state": trainer_state,
        "data_cursor": cursor,
    }


def build_preflight_report(
    configs: Mapping[str, ModelConfig],
    *,
    config_paths: Mapping[str, Path],
    batch_size: int,
    context_length: int | None,
    run_cpu_smoke: bool,
    stop_criteria: ScaleStopCriteria,
) -> dict[str, object]:
    models: dict[str, object] = {}
    for name, config in configs.items():
        estimates = estimate_scale(
            config,
            batch_size=batch_size,
            context_length=context_length,
        )
        models[name] = {
            "config": config.to_dict(),
            "config_sha256": _sha256(config_paths[name]),
            "estimates": asdict(estimates),
            "cuda": {
                "status": "unmeasured",
                "peak_allocated_vram_bytes": "unmeasured",
                "peak_reserved_vram_bytes": "unmeasured",
                "training_tokens_per_second": "unmeasured",
                "decode_tokens_per_second": "unmeasured",
            },
        }
    return {
        "schema_version": 1,
        "status": "preflight_only",
        "quality_claim": False,
        "scale_decision": "unmeasured",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "batch_size": batch_size,
        "context_length": context_length or "config_default",
        "stop_criteria": asdict(stop_criteria),
        "models": models,
        "cpu_checkpoint_smoke": cpu_checkpoint_smoke() if run_cpu_smoke else "not_run",
    }


def write_json_report(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def decide_500m(
    measurement: Mapping[str, Any],
    *,
    stop_criteria: ScaleStopCriteria,
    expected_learning_value: Literal["low", "medium", "high"],
) -> dict[str, object]:
    """Return a real-data decision, refusing CPU, estimated, incomplete, or failed evidence."""

    required = {
        "device_type",
        "status",
        "peak_reserved_vram_fraction",
        "wall_hours",
        "oom_failures",
        "nonfinite_updates",
        "completed_updates",
        "training_tokens_per_second",
        "validation_loss_start",
        "validation_loss_end",
        "checkpoint_resume_passed",
    }
    missing = sorted(required - set(measurement))
    if missing:
        raise ValueError(f"scale measurement is missing required fields: {missing}")
    if measurement["device_type"] != "cuda" or measurement["status"] != "completed":
        raise ValueError("500M decision requires a completed real CUDA measurement")
    numeric_fields = required - {"device_type", "status", "checkpoint_resume_passed"}
    if any(measurement[field] == "unmeasured" for field in numeric_fields):
        raise ValueError("500M decision cannot consume unmeasured fields")
    reasons: list[str] = []
    if float(measurement["peak_reserved_vram_fraction"]) > stop_criteria.max_peak_vram_fraction:
        reasons.append("peak_reserved_vram_fraction")
    if float(measurement["wall_hours"]) > stop_criteria.max_wall_hours:
        reasons.append("wall_hours")
    if int(measurement["oom_failures"]) > stop_criteria.max_oom_failures:
        reasons.append("oom_failures")
    if int(measurement["nonfinite_updates"]) > stop_criteria.max_nonfinite_updates:
        reasons.append("nonfinite_updates")
    if int(measurement["completed_updates"]) < stop_criteria.min_training_updates:
        reasons.append("insufficient_updates")
    if not bool(measurement["checkpoint_resume_passed"]):
        reasons.append("checkpoint_resume")
    if float(measurement["validation_loss_end"]) >= float(measurement["validation_loss_start"]):
        reasons.append("no_validation_improvement")
    if float(measurement["training_tokens_per_second"]) <= 0:
        reasons.append("nonpositive_throughput")
    if expected_learning_value == "low":
        reasons.append("low_expected_learning_value")
    decision = "go" if not reasons else "no_go"
    return {
        "schema_version": 1,
        "decision": decision,
        "expected_learning_value": expected_learning_value,
        "reasons": reasons,
        "stop_criteria": asdict(stop_criteria),
        "source_measurement": dict(measurement),
        "v1_artifacts_unchanged": True,
    }


def assemble_cuda_scale_measurement(
    training_run: Mapping[str, Any],
    inference_run: Mapping[str, Any],
    *,
    validation_loss_start: float,
    validation_loss_end: float,
    checkpoint_resume_passed: bool,
    oom_failures: int,
    nonfinite_updates: int,
    projected_full_run_hours: float,
    projected_cost: float,
    cost_currency: str,
) -> dict[str, object]:
    """Merge actual trainer/profiler records; refuse CPU or incomplete synthetic evidence."""

    if inference_run.get("status") != "completed" or inference_run.get("device_type") != "cuda":
        raise ValueError("inference evidence must be a completed CUDA profile")
    if str(training_run.get("device", "")).casefold() == "cpu":
        raise ValueError("training evidence must come from CUDA hardware")
    if validation_loss_start <= 0 or validation_loss_end <= 0:
        raise ValueError("validation losses must be positive measured values")
    if min(oom_failures, nonfinite_updates) < 0:
        raise ValueError("failure counts cannot be negative")
    if projected_full_run_hours <= 0 or projected_cost < 0 or not cost_currency:
        raise ValueError("projected time/cost fields are invalid")
    metrics = dict(training_run.get("metrics", {}))
    required_training = (
        "completed_updates",
        "peak_reserved_vram_bytes",
        "total_vram_bytes",
    )
    if any(field not in metrics for field in required_training):
        raise ValueError("training run lacks CUDA update/VRAM metrics")
    total_vram = float(metrics["total_vram_bytes"])
    if total_vram <= 0:
        raise ValueError("training run total_vram_bytes must be measured")
    peak_reserved = max(
        float(metrics["peak_reserved_vram_bytes"]),
        float(inference_run["peak_reserved_vram_bytes"]),
    )
    throughput = float(training_run.get("tokens_per_second", 0))
    wall_seconds = float(training_run.get("wall_seconds", 0))
    if throughput <= 0 or wall_seconds <= 0:
        raise ValueError("training throughput and wall time must be measured and positive")
    return {
        "schema_version": 1,
        "device_type": "cuda",
        "device": training_run.get("device"),
        "status": "completed",
        "peak_reserved_vram_fraction": peak_reserved / total_vram,
        "peak_reserved_vram_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "wall_hours": wall_seconds / 3600,
        "projected_full_run_hours": projected_full_run_hours,
        "projected_cost": projected_cost,
        "cost_currency": cost_currency,
        "oom_failures": oom_failures,
        "nonfinite_updates": nonfinite_updates,
        "completed_updates": int(metrics["completed_updates"]),
        "training_tokens_per_second": throughput,
        "validation_loss_start": validation_loss_start,
        "validation_loss_end": validation_loss_end,
        "checkpoint_resume_passed": checkpoint_resume_passed,
        "inference": dict(inference_run),
        "source_training_run": dict(training_run),
    }
