"""Train and verify a MiniFrontier byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from minifrontier.tokenizer import VOCAB_SIZE, train_byte_bpe


def iter_input_text(paths: list[Path], *, jsonl_field: str) -> Iterator[str]:
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    value = row.get(jsonl_field)
                    if not isinstance(value, str):
                        raise ValueError(f"{path}:{line_number} lacks string field {jsonl_field!r}")
                    yield value
        else:
            yield path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--jsonl-field", default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = train_byte_bpe(
        iter_input_text(args.inputs, jsonl_field=args.jsonl_field),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(args.output)
    print(f"saved tokenizer with {tokenizer.vocab_size:,} entries to {args.output}")


if __name__ == "__main__":
    main()
