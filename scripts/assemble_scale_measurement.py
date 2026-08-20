"""Merge real CUDA trainer/profile outputs into the MF-070 decision input schema."""

# Gathers what actually happened on real GPU runs -- throughput, memory, wall time
# -- into the single structured record the scale decision reads. Plumbing between
# "we measured things" and "we decided something", kept separate so the decision
# cannot quietly invent its own inputs.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.scale import assemble_cuda_scale_measurement, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--inference-run", type=Path, required=True)
    parser.add_argument("--validation-loss-start", type=float, required=True)
    parser.add_argument("--validation-loss-end", type=float, required=True)
    parser.add_argument("--checkpoint-resume-passed", action="store_true")
    parser.add_argument("--oom-failures", type=int, default=0)
    parser.add_argument("--nonfinite-updates", type=int, default=0)
    parser.add_argument("--projected-full-run-hours", type=float, required=True)
    parser.add_argument("--projected-cost", type=float, required=True)
    parser.add_argument("--cost-currency", default="USD")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = assemble_cuda_scale_measurement(
        json.loads(args.training_run.read_text(encoding="utf-8")),
        json.loads(args.inference_run.read_text(encoding="utf-8")),
        validation_loss_start=args.validation_loss_start,
        validation_loss_end=args.validation_loss_end,
        checkpoint_resume_passed=args.checkpoint_resume_passed,
        oom_failures=args.oom_failures,
        nonfinite_updates=args.nonfinite_updates,
        projected_full_run_hours=args.projected_full_run_hours,
        projected_cost=args.projected_cost,
        cost_currency=args.cost_currency,
    )
    write_json_report(args.output, report)
    print(f"wrote measured CUDA scale record to {args.output}")


if __name__ == "__main__":
    main()
