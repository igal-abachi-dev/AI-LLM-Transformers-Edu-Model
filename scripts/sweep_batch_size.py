"""MF-101: real batch-size x gradient-accumulation throughput/VRAM sweep.

Runs train/pretrain.py once per (--batch-size, --accumulation-steps)
combination via --no-checkpoint (no periodic/final checkpoint files, just
run.json -- this is a throughput measurement, not a quality comparison), and
aggregates each arm's real tokens_per_second/peak_memory_mb into one report.
No new estimation logic: every number here is read directly from a real
run.json train/pretrain.py already writes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--accumulation-steps", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, default=Path("reports/mf101-batch-size-sweep.json"))
    return parser.parse_args()


def run_arm(
    *,
    config: Path,
    train_shards: Path,
    output: Path,
    batch_size: int,
    accumulation_steps: int,
    updates: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "train/pretrain.py",
            "--config",
            str(config),
            "--train-shards",
            str(train_shards),
            "--output",
            str(output),
            "--updates",
            str(updates),
            "--batch-size",
            str(batch_size),
            "--accumulation-steps",
            str(accumulation_steps),
            "--seed",
            str(seed),
            "--device",
            device,
            "--no-checkpoint",
        ],
        check=True,
    )
    return json.loads((output / "run.json").read_text(encoding="utf-8"))


def summarize_arm(
    *, batch_size: int, accumulation_steps: int, run_metadata: dict[str, Any]
) -> dict[str, Any]:
    completed_updates = run_metadata["metrics"]["completed_updates"]
    if completed_updates <= 0:
        raise ValueError("run produced zero completed updates")
    return {
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch": batch_size * accumulation_steps,
        "tokens_per_update": run_metadata["train_tokens"] / completed_updates,
        "tokens_per_second": run_metadata["tokens_per_second"],
        "peak_memory_mb": run_metadata["peak_memory_mb"],
        "train_loss": run_metadata["train_loss"],
    }


def main() -> None:
    args = parse_args()
    results = []
    for batch_size in args.batch_sizes:
        for accumulation_steps in args.accumulation_steps:
            label = f"b{batch_size}-a{accumulation_steps}"
            run_metadata = run_arm(
                config=args.config,
                train_shards=args.train_shards,
                output=args.output_dir / label,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                updates=args.updates,
                seed=args.seed,
                device=args.device,
            )
            summary = summarize_arm(
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                run_metadata=run_metadata,
            )
            results.append(summary)
            print(json.dumps(summary))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"config": str(args.config), "updates": args.updates, "arms": results}
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
