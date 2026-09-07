"""MF-090: cheap, no-GPU fertility (bytes/token) triage across tokenizer candidates.

Fertility is a static property of a tokenizer plus a text sample -- no model
training required. Before spending real GPU hours on a full retrain+repack+
train+validate comparison for every candidate, this measures which mechanism
(vocabulary size, digit-splitting, or the digit-split pattern's leading-space
handling) is actually responsible for MF-087's real negative BPB result, using
a genuinely held-out text sample (decoded back from an existing real
validation shard, never part of any tokenizer's training corpus).

This does not replace a real trained-model BPB comparison -- fertility and
model quality are correlated but not identical (a tokenizer can look more
efficient in raw bytes/token while still being harder for a fixed-size model
to learn). It exists to narrow which candidates deserve that more expensive
comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.shards import PackedShardDataset
from minifrontier.tokenizer import MiniFrontierTokenizer


def decode_held_out_text(
    reference_tokenizer: MiniFrontierTokenizer,
    validation_shards: Path,
    *,
    max_sequences: int | None,
) -> str:
    """Decode real validation sequences back to text, skipping padding."""

    dataset = PackedShardDataset(validation_shards)
    limit = len(dataset) if max_sequences is None else min(max_sequences, len(dataset))
    pieces = []
    for index in range(limit):
        tokens, non_padding = dataset[index]
        ids = tokens[:non_padding].tolist()
        pieces.append(reference_tokenizer.decode(ids, skip_special_tokens=True))
    return "\n".join(pieces)


def measure_fertility(tokenizer: MiniFrontierTokenizer, text: str) -> dict[str, float | int]:
    token_count = len(tokenizer.encode(text))
    byte_count = len(text.encode("utf-8"))
    if token_count == 0:
        raise ValueError("held-out text produced zero tokens -- nothing to measure")
    return {
        "tokens": token_count,
        "utf8_bytes": byte_count,
        "bytes_per_token": byte_count / token_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-tokenizer", type=Path, required=True)
    parser.add_argument("--reference-shards", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable; each candidate tokenizer to measure, e.g. --tokenizer 32k=data/tokenizer",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_tokenizer = MiniFrontierTokenizer.from_directory(args.reference_tokenizer)
    text = decode_held_out_text(
        reference_tokenizer, args.reference_shards, max_sequences=args.max_sequences
    )
    held_out_bytes = len(text.encode("utf-8"))
    print(f"held-out text: {held_out_bytes:,} UTF-8 bytes, decoded via {args.reference_tokenizer}")

    results = {}
    for spec in args.tokenizer:
        label, _, path = spec.partition("=")
        if not label or not path:
            raise ValueError(f"--tokenizer must be LABEL=PATH, got {spec!r}")
        tokenizer = MiniFrontierTokenizer.from_directory(Path(path))
        results[label] = {
            "path": path,
            "vocab_size": tokenizer.vocab_size,
            **measure_fertility(tokenizer, text),
        }
        print(
            f"{label:>24s}  vocab={tokenizer.vocab_size:>6d}  "
            f"tokens={results[label]['tokens']:>10,d}  "
            f"bytes/token={results[label]['bytes_per_token']:.4f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "reference_tokenizer": str(args.reference_tokenizer),
                    "reference_shards": str(args.reference_shards),
                    "held_out_utf8_bytes": len(text.encode("utf-8")),
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
