from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from minifrontier.config import ModelConfig
from minifrontier.scale import (
    ScaleStopCriteria,
    assemble_cuda_scale_measurement,
    build_preflight_report,
    cpu_checkpoint_smoke,
    decide_500m,
    estimate_scale,
    exact_parameter_count,
)


def test_meta_parameter_count_matches_real_tiny_model() -> None:
    config = ModelConfig.tiny_modern(attention_impl="sdpa")
    from minifrontier.model import MiniFrontier

    assert exact_parameter_count(config) == MiniFrontier(config).parameter_count()


def test_scale_estimates_label_lower_bounds_and_bounded_cache_savings() -> None:
    config = ModelConfig.from_toml("configs/350m-modern.toml")
    estimates = estimate_scale(config, batch_size=1)
    assert estimates.parameter_count == 332_460_544
    assert estimates.training_lower_bound_bytes == estimates.parameter_count * 12
    assert estimates.bounded_local_kv_bytes < estimates.full_history_kv_bytes
    assert any("lower bounds" in value for value in estimates.assumptions)


def test_preflight_never_invents_cuda_or_scale_decision(tmp_path: Path) -> None:
    path = Path("configs/350m-modern.toml")
    report = build_preflight_report(
        {"350m-modern": ModelConfig.from_toml(path)},
        config_paths={"350m-modern": path},
        batch_size=1,
        context_length=512,
        run_cpu_smoke=False,
        stop_criteria=ScaleStopCriteria(),
    )
    assert report["status"] == "preflight_only"
    assert report["scale_decision"] == "unmeasured"
    model = report["models"]["350m-modern"]
    assert model["cuda"]["training_tokens_per_second"] == "unmeasured"


def test_tiny_cpu_checkpoint_smoke_passes() -> None:
    report = cpu_checkpoint_smoke()
    assert report["status"] == "passed"
    assert report["checkpoint_logits_exact"] is True
    assert report["data_cursor"] == {"cursor": 1}


def _measurement(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "device_type": "cuda",
        "status": "completed",
        "peak_reserved_vram_fraction": 0.8,
        "wall_hours": 1.0,
        "oom_failures": 0,
        "nonfinite_updates": 0,
        "completed_updates": 20,
        "training_tokens_per_second": 100.0,
        "validation_loss_start": 5.0,
        "validation_loss_end": 4.5,
        "checkpoint_resume_passed": True,
    }
    values.update(changes)
    return values


def test_scale_decision_requires_real_complete_cuda_measurement() -> None:
    criteria = ScaleStopCriteria()
    with pytest.raises(ValueError, match="CUDA"):
        decide_500m(
            _measurement(device_type="cpu"),
            stop_criteria=criteria,
            expected_learning_value="high",
        )
    report = decide_500m(
        _measurement(),
        stop_criteria=criteria,
        expected_learning_value="high",
    )
    assert report["decision"] == "go"
    no_go = decide_500m(
        _measurement(peak_reserved_vram_fraction=0.99),
        stop_criteria=criteria,
        expected_learning_value="high",
    )
    assert no_go["decision"] == "no_go"
    assert "peak_reserved_vram_fraction" in no_go["reasons"]
    assert asdict(criteria) == no_go["stop_criteria"]


def test_cuda_measurement_assembly_rejects_cpu_and_preserves_cost() -> None:
    training = {
        "device": "NVIDIA RTX",
        "wall_seconds": 360,
        "tokens_per_second": 1000,
        "metrics": {
            "completed_updates": 20,
            "peak_reserved_vram_bytes": 8_000,
            "total_vram_bytes": 10_000,
        },
    }
    inference = {
        "status": "completed",
        "device_type": "cuda",
        "peak_reserved_vram_bytes": 7_000,
    }
    report = assemble_cuda_scale_measurement(
        training,
        inference,
        validation_loss_start=5.0,
        validation_loss_end=4.0,
        checkpoint_resume_passed=True,
        oom_failures=0,
        nonfinite_updates=0,
        projected_full_run_hours=12,
        projected_cost=5,
        cost_currency="USD",
    )
    assert report["peak_reserved_vram_fraction"] == 0.8
    assert report["projected_cost"] == 5
    with pytest.raises(ValueError, match="CUDA"):
        assemble_cuda_scale_measurement(
            {**training, "device": "CPU"},
            inference,
            validation_loss_start=5.0,
            validation_loss_end=4.0,
            checkpoint_resume_passed=True,
            oom_failures=0,
            nonfinite_updates=0,
            projected_full_run_hours=12,
            projected_cost=5,
            cost_currency="USD",
        )
