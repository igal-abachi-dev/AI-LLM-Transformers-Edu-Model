"""Create a matched Edu-versus-Modern report from comparable benchmark records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from minifrontier.evaluation.benchmark import read_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edu", type=Path, required=True)
    parser.add_argument("--modern", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed_count <= 0:
        raise ValueError("seed_count must be positive")
    edu = read_record(args.edu)
    modern = read_record(args.modern)
    if not edu.comparable_to(modern):
        raise ValueError("Edu and Modern records do not share the frozen comparison key")
    common_metrics = set(edu.metrics) & set(modern.metrics)
    deltas = {name: modern.metrics[name] - edu.metrics[name] for name in common_metrics}
    report = {
        "status": "matched_teaching_comparison",
        "comparison_key": asdict(edu.comparison_key),
        "seed_count": args.seed_count,
        "claim_strength": "causal" if args.seed_count >= 3 else "exploratory",
        "edu": asdict(edu),
        "modern": asdict(modern),
        "modern_minus_edu": deltas,
        "limitations": [
            "Fewer than three seeds permits exploratory, not strong causal, conclusions."
            if args.seed_count < 3
            else "Uncertainty must still be reported across the recorded seeds."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
