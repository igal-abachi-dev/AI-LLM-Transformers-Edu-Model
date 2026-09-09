from __future__ import annotations

import pytest

from scripts.sweep_batch_size import summarize_arm, updates_for_token_budget


def test_updates_for_token_budget_scales_down_as_effective_batch_grows() -> None:
    cheap = updates_for_token_budget(
        target_tokens=400_000,
        batch_size=2,
        accumulation_steps=1,
        sequence_length=1024,
        min_updates=10,
    )
    expensive = updates_for_token_budget(
        target_tokens=400_000,
        batch_size=8,
        accumulation_steps=32,
        sequence_length=1024,
        min_updates=10,
    )
    assert cheap > expensive
    assert cheap == max(10, round(400_000 / (2 * 1 * 1024)))


def test_updates_for_token_budget_never_drops_below_the_floor() -> None:
    updates = updates_for_token_budget(
        target_tokens=1, batch_size=8, accumulation_steps=32, sequence_length=1024, min_updates=10
    )
    assert updates == 10


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
