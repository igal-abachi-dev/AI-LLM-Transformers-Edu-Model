"""Validate measured 350M CUDA evidence and write a fail-closed 500M go/no-go record."""

# Decides whether training the next size up is worth it, from measured evidence
# only. "Fail-closed" means a missing measurement produces a refusal, never an
# optimistic guess -- the default answer is no, and evidence has to move it.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.scale import ScaleStopCriteria, decide_500m, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-learning-value", choices=("low", "medium", "high"), required=True
    )
    parser.add_argument("--max-peak-vram-fraction", type=float, default=0.95)
    parser.add_argument("--max-wall-hours", type=float, default=24.0)
    parser.add_argument("--max-oom-failures", type=int, default=1)
    parser.add_argument("--max-nonfinite-updates", type=int, default=0)
    parser.add_argument("--min-training-updates", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measurement = json.loads(args.measurement.read_text(encoding="utf-8"))
    criteria = ScaleStopCriteria(
        max_peak_vram_fraction=args.max_peak_vram_fraction,
        max_wall_hours=args.max_wall_hours,
        max_oom_failures=args.max_oom_failures,
        max_nonfinite_updates=args.max_nonfinite_updates,
        min_training_updates=args.min_training_updates,
    )
    report = decide_500m(
        measurement,
        stop_criteria=criteria,
        expected_learning_value=args.expected_learning_value,
    )
    write_json_report(args.output, report)
    print(f"wrote 500M decision={report['decision']} to {args.output}")


if __name__ == "__main__":
    main()
