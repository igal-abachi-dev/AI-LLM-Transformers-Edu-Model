"""MF-086: run the real synthetic needle-in-haystack retrieval eval against a
release, or (MF-082) directly against a raw training checkpoint from a bounded
comparison run -- going through a full `export_release` just to evaluate a
throwaway 5,000-update comparison arm would be needless ceremony."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.checkpoint import load_release, load_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.evaluation.retrieval import run_needle_haystack_eval
from minifrontier.model import MiniFrontier
from minifrontier.tokenizer import MiniFrontierTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release", type=Path, help="a published release directory")
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="a raw train/pretrain.py checkpoint directory (e.g. .../final), "
        "for bounded comparisons that never go through export_release",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizer"),
        help="only used with --checkpoint; --release carries its own tokenizer",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument(
        "--needle-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_model(args: argparse.Namespace) -> tuple[MiniFrontier, MiniFrontierTokenizer]:
    if args.release is not None:
        return load_release(args.release, device=args.device)
    config = ModelConfig(
        **json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    model = MiniFrontier(config).to(args.device).eval()
    load_training_checkpoint(args.checkpoint, model, trusted_local_state=False)
    tokenizer = MiniFrontierTokenizer.from_directory(args.tokenizer)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    model, tokenizer = _load_model(args)
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
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
