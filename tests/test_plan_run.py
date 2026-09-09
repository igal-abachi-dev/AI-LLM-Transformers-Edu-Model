from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plan_run import estimate_run, tokens_per_second_from_run_json


def test_estimate_run_matches_hand_arithmetic() -> None:
    # 3,000,000,000 tokens at 2 * 1024 tokens/update, 4200 tok/s -- the same
    # shape of arithmetic reports/mf070-* and docs/IMPLEMENTATION_DECISIONS.md
    # have repeated by hand all session.
    result = estimate_run(
        target_tokens=3_000_000_000,
        tokens_per_second=4200.0,
        batch_size=2,
        sequence_length=1024,
        gradient_accumulation_steps=1,
    )
    assert result["tokens_per_update"] == 2048
    assert result["estimated_updates"] == 1_464_844  # ceil(3e9 / 2048)
    seconds = result["estimated_wall_seconds"]
    assert seconds == pytest.approx(3_000_000_000 / 4200.0)
    assert result["estimated_wall_hours"] == pytest.approx(seconds / 3600)
    assert result["estimated_calendar_days"] == pytest.approx(seconds / 86400)


def test_estimate_run_accounts_for_gradient_accumulation() -> None:
    result = estimate_run(
        target_tokens=1_000_000,
        tokens_per_second=1000.0,
        batch_size=2,
        sequence_length=512,
        gradient_accumulation_steps=4,
    )
    assert result["tokens_per_update"] == 2 * 512 * 4


@pytest.mark.parametrize(
    ("target_tokens", "tokens_per_second", "batch_size", "sequence_length", "grad_accum"),
    [
        (0, 1.0, 1, 1, 1),
        (1, 0.0, 1, 1, 1),
        (1, 1.0, 0, 1, 1),
        (1, 1.0, 1, 0, 1),
        (1, 1.0, 1, 1, 0),
    ],
)
def test_estimate_run_rejects_non_positive_inputs(
    target_tokens, tokens_per_second, batch_size, sequence_length, grad_accum
) -> None:
    with pytest.raises(ValueError):
        estimate_run(
            target_tokens=target_tokens,
            tokens_per_second=tokens_per_second,
            batch_size=batch_size,
            sequence_length=sequence_length,
            gradient_accumulation_steps=grad_accum,
        )


def test_tokens_per_second_from_run_json_reads_real_field(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps({"tokens_per_second": 4001.9, "train_tokens": 10_230_000}), encoding="utf-8"
    )
    assert tokens_per_second_from_run_json(run_json) == pytest.approx(4001.9)


def test_tokens_per_second_from_run_json_rejects_missing_field(tmp_path: Path) -> None:
    trainer_state = tmp_path / "trainer_state.json"
    trainer_state.write_text(json.dumps({"training_state": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="tokens_per_second"):
        tokens_per_second_from_run_json(trainer_state)
