"""MF-101: real batch-size x gradient-accumulation throughput/VRAM sweep.

Runs train/pretrain.py once per (--batch-size, --accumulation-steps)
combination via --no-checkpoint (no periodic/final checkpoint files, just
run.json -- this is a throughput measurement, not a quality comparison), and
aggregates each arm's real tokens_per_second/peak_memory_mb into one report.
No new estimation logic: every number here is read directly from a real
run.json train/pretrain.py already writes.

Updates-per-arm is derived from a fixed *token* budget, not a fixed update
count: accumulation_steps multiplies real compute per update (an update at
batch=8/accum=32 processes 128x the tokens of one at batch=2/accum=1), so a
uniform update count across every combination would make the
high-accumulation arms take drastically, impractically longer than the
others for no measurement benefit -- verified directly: 20 updates at
batch=8/accum=32 on 150m-modern ran over 20 minutes without finishing.
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
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument(
        "--target-tokens-per-arm",
        type=int,
        default=400_000,
        help="real per-arm token budget; updates = target / (batch*accum*sequence_length)",
    )
    parser.add_argument("--min-updates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, default=Path("reports/mf101-batch-size-sweep.json"))
    return parser.parse_args()


def updates_for_token_budget(
    *,
    target_tokens: int,
    batch_size: int,
    accumulation_steps: int,
    sequence_length: int,
    min_updates: int,
) -> int:
    tokens_per_update = batch_size * accumulation_steps * sequence_length
    return max(min_updates, round(target_tokens / tokens_per_update))


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
    warmup = max(1, min(updates // 10, 5))
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
            "--warmup-updates",
            str(warmup),
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
            updates = updates_for_token_budget(
                target_tokens=args.target_tokens_per_arm,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                sequence_length=args.sequence_length,
                min_updates=args.min_updates,
            )
            run_metadata = run_arm(
                config=args.config,
                train_shards=args.train_shards,
                output=args.output_dir / label,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                updates=updates,
                seed=args.seed,
                device=args.device,
            )
            summary = summarize_arm(
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                run_metadata=run_metadata,
            )
            summary["updates"] = updates
            results.append(summary)
            print(json.dumps(summary))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": str(args.config),
        "target_tokens_per_arm": args.target_tokens_per_arm,
        "arms": results,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
