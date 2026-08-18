"""Create a validated draft or evidence-backed frozen V1 training protocol."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from minifrontier.release import TrainingProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", choices=("draft", "frozen"), default="draft")
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--data-mixture-id", required=True)
    parser.add_argument("--target-tokens", type=int, default=3_000_000_000)
    parser.add_argument("--batch-tokens", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--evaluation-interval", type=int, required=True)
    parser.add_argument("--evidence", type=Path, nargs="*", default=[])
    parser.add_argument("--approve-undertrained", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    evidence = {str(path): file_sha256(path) for path in args.evidence}
    protocol = TrainingProtocol(
        status=args.status,
        tokenizer_sha256=args.tokenizer_sha256,
        data_mixture_id=args.data_mixture_id,
        target_tokens=args.target_tokens,
        batch_tokens=args.batch_tokens,
        sequence_length=args.sequence_length,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        seeds=tuple(args.seeds),
        evaluation_interval=args.evaluation_interval,
        evidence_sha256=evidence,
        undertrained_approved=args.approve_undertrained,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol.write(args.output)
    print(f"wrote {args.status} protocol to {args.output}")


if __name__ == "__main__":
    main()
