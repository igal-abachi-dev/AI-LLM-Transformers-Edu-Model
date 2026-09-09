"""Estimate updates/wall-clock/calendar days for a token budget (MF-099).

Replaces the ad hoc inline arithmetic this project's own reports have
repeated by hand all session (e.g. the 3B-token-target reconsideration in
`docs/IMPLEMENTATION_DECISIONS.md`) with one small, reusable tool. No new
estimation logic: `tokens_per_update = batch_size * sequence_length *
gradient_accumulation_steps`, `updates = ceil(target_tokens /
tokens_per_update)`, `wall_seconds = target_tokens / tokens_per_second` --
the same arithmetic every prior report already computed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from minifrontier.config import ModelConfig
from minifrontier.scale import exact_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="defaults to the config's max_seq_len",
    )
    throughput = parser.add_mutually_exclusive_group(required=True)
    throughput.add_argument("--tokens-per-second", type=float)
    throughput.add_argument(
        "--from-run-json",
        type=Path,
        help="read a real measured tokens_per_second from an existing run.json",
    )
    return parser.parse_args()


def tokens_per_second_from_run_json(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "tokens_per_second" not in data:
        raise ValueError(
            f"{path} has no 'tokens_per_second' field -- point --from-run-json at a "
            "real run.json (trainer_state.json alone does not record wall-clock time)"
        )
    return float(data["tokens_per_second"])


def estimate_run(
    *,
    target_tokens: int,
    tokens_per_second: float,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    if batch_size <= 0 or sequence_length <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError(
            "batch_size, sequence_length, and gradient_accumulation_steps must be positive"
        )
    tokens_per_update = batch_size * sequence_length * gradient_accumulation_steps
    wall_seconds = target_tokens / tokens_per_second
    return {
        "target_tokens": target_tokens,
        "tokens_per_update": tokens_per_update,
        "estimated_updates": math.ceil(target_tokens / tokens_per_update),
        "tokens_per_second": tokens_per_second,
        "estimated_wall_seconds": wall_seconds,
        "estimated_wall_hours": wall_seconds / 3600,
        "estimated_calendar_days": wall_seconds / 86400,
    }


def main() -> None:
    args = parse_args()
    config = ModelConfig.from_toml(args.config)
    sequence_length = args.sequence_length or config.max_seq_len
    tokens_per_second = (
        args.tokens_per_second
        if args.tokens_per_second is not None
        else tokens_per_second_from_run_json(args.from_run_json)
    )
    estimate = estimate_run(
        target_tokens=args.target_tokens,
        tokens_per_second=tokens_per_second,
        batch_size=args.batch_size,
        sequence_length=sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    estimate["config"] = str(args.config)
    estimate["parameter_count"] = exact_parameter_count(config)
    print(json.dumps(estimate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
