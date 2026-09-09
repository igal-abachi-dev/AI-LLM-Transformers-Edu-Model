from __future__ import annotations

import pytest

from scripts.sweep_batch_size import summarize_arm


def _run_metadata(**overrides) -> dict:
    base = {
        "train_tokens": 409_600,
        "tokens_per_second": 4200.5,
        "peak_memory_mb": 5760.0,
        "train_loss": 4.5,
        "metrics": {"completed_updates": 200.0},
    }
    base.update(overrides)
    return base


def test_summarize_arm_computes_real_fields_from_run_metadata() -> None:
    summary = summarize_arm(batch_size=2, accumulation_steps=8, run_metadata=_run_metadata())
    assert summary["batch_size"] == 2
    assert summary["accumulation_steps"] == 8
    assert summary["effective_batch"] == 16
    assert summary["tokens_per_update"] == pytest.approx(409_600 / 200)
    assert summary["tokens_per_second"] == 4200.5
    assert summary["peak_memory_mb"] == 5760.0
    assert summary["train_loss"] == 4.5


def test_summarize_arm_rejects_zero_completed_updates() -> None:
    metadata = _run_metadata(metrics={"completed_updates": 0.0})
    with pytest.raises(ValueError, match="zero completed updates"):
        summarize_arm(batch_size=2, accumulation_steps=1, run_metadata=metadata)
