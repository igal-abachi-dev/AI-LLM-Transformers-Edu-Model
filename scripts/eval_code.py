"""Score versioned code/FIM predictions with explicit contamination metadata."""

# Scores generated code, and records what the score can and cannot mean.
#
# "Contamination metadata" is the important part. Public coding benchmarks have
# long since leaked into web-crawled training data, so a model may have read the
# answers -- in which case a high score measures memorization, not ability. The
# fixtures here are original to this project for that reason, and the provenance
# is recorded alongside the number so nobody has to guess later.

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from minifrontier.evaluation.code import (
    contamination_report,
    load_fixtures,
    score_fixture_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--training-signatures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-trusted-fixtures", action="store_true")
    parser.add_argument(
        "--general-lm-metrics",
        type=Path,
        help="Optional matched baseline/FIM language-metric JSON included in the report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures = load_fixtures(args.fixtures)
    predictions = {
        str(row["id"]): str(row["prediction"])
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    signatures = {"exact": [], "simhash": []}
    if args.training_signatures is not None:
        signatures = json.loads(args.training_signatures.read_text(encoding="utf-8"))
    contamination = contamination_report(
        fixtures,
        training_hashes=set(signatures["exact"]),
        training_simhashes={int(value, 16) for value in signatures["simhash"]},
    )
    scores = score_fixture_predictions(
        fixtures,
        predictions,
        execute_trusted_fixtures=args.execute_trusted_fixtures,
    )
    total = len(scores)
    if total == 0:
        raise ValueError("at least one evaluation fixture is required")
    functional = [score.functional for score in scores if score.functional is not None]
    general_lm = (
        json.loads(args.general_lm_metrics.read_text(encoding="utf-8"))
        if args.general_lm_metrics is not None
        else {"status": "not_provided", "quality_claim": False}
    )
    report = {
        "fixture_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "contamination": asdict(contamination),
        "metrics": {
            "count": total,
            "exact_rate": sum(score.exact for score in scores) / total,
            "syntax_valid_rate": sum(score.syntax_valid for score in scores) / total,
            "compile_rate": sum(score.compiles for score in scores) / total,
            "functional_count": len(functional),
            "functional_rate": (
                sum(value is True for value in functional) / len(functional) if functional else None
            ),
        },
        "general_lm_regression": general_lm,
        "results": [asdict(score) for score in scores],
        "limitations": [
            "Small deterministic fixtures measure pipeline correctness, not broad coding quality.",
            "Functional execution is allowed only for repository-owned trusted fixtures.",
            "Report variance across training seeds before making an effect claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
