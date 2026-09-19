"""CLI wiring tests for scripts/eval_code.py, including MF-148's pass@k mode.

The scoring logic itself (score_fixture_predictions, score_fixture_predictions_pass_at_k,
pass_at_k) is unit-tested directly in tests/test_evaluation.py -- these tests only prove
the CLI reads/writes the right files and dispatches to the right report shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.eval_code as eval_code


def _write_fixture(path: Path) -> None:
    fixture = {
        "id": "add-1",
        "kind": "completion",
        "language": "python",
        "prompt": "def add(a, b):\n    ",
        "reference": "return a + b\n",
        "tests": "assert add(2, 3) == 5",
    }
    path.write_text(json.dumps(fixture) + "\n", encoding="utf-8")


def test_eval_code_default_single_sample_path_writes_the_original_report_shape(
    tmp_path, monkeypatch
) -> None:
    fixtures_path = tmp_path / "fixtures.jsonl"
    _write_fixture(fixtures_path)
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "add-1", "prediction": "return a + b\n"}) + "\n", encoding="utf-8"
    )
    output_path = tmp_path / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_code.py",
            "--fixtures",
            str(fixtures_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
            "--execute-trusted-fixtures",
        ],
    )
    eval_code.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["metrics"]["count"] == 1
    assert report["metrics"]["exact_rate"] == 1.0
    assert report["metrics"]["functional_rate"] == 1.0
    assert "mean_pass_at_k" not in report["metrics"]


def test_eval_code_pass_at_k_flag_writes_the_pass_at_k_report_shape(tmp_path, monkeypatch) -> None:
    fixtures_path = tmp_path / "fixtures.jsonl"
    _write_fixture(fixtures_path)
    predictions_path = tmp_path / "predictions.jsonl"
    # 3 samples: 2 correct, 1 wrong (compiles, fails the test) -> n=3, c=2.
    row = {
        "id": "add-1",
        "predictions": ["return a + b\n", "return a + b\n", "return a - b\n"],
    }
    predictions_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_path = tmp_path / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_code.py",
            "--fixtures",
            str(fixtures_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
            "--execute-trusted-fixtures",
            "--pass-at-k",
            "1",
            "--pass-at-k",
            "2",
        ],
    )
    eval_code.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["metrics"]["count"] == 1
    assert set(report["metrics"]["mean_pass_at_k"]) == {"1", "2"}
    assert report["results"][0]["num_samples"] == 3
    assert report["results"][0]["num_functional_passes"] == 2
    assert "exact_rate" not in report["metrics"]
