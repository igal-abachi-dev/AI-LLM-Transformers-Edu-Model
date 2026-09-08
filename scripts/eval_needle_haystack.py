"""MF-086: run the real synthetic needle-in-haystack retrieval eval against a release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.checkpoint import load_release
from minifrontier.evaluation.retrieval import run_needle_haystack_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument(
        "--needle-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_release(args.release, device=args.device)
    trials = run_needle_haystack_eval(
        model,
        tokenizer,
        context_lengths=args.context_lengths,
        needle_fractions=args.needle_fractions,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    by_length: dict[int, list[dict[str, object]]] = {}
    for trial in trials:
        by_length.setdefault(trial.context_length, []).append(
            {
                "needle_fraction": trial.needle_fraction,
                "found": trial.found,
                "code": trial.code,
                "completion": trial.completion,
            }
        )
        print(
            f"length={trial.context_length:>5} fraction={trial.needle_fraction:.2f} "
            f"found={trial.found} code={trial.code} completion={trial.completion!r}"
        )
    summary = {
        length: sum(t["found"] for t in items) / len(items) for length, items in by_length.items()
    }
    print("retrieval rate by context length:", summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "release": str(args.release),
            "context_lengths": args.context_lengths,
            "needle_fractions": args.needle_fractions,
            "seed": args.seed,
            "trials": by_length,
            "retrieval_rate_by_context_length": summary,
        }
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
