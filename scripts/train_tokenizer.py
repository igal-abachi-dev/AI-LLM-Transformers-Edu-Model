"""Train and verify a MiniFrontier byte-level BPE tokenizer."""

# STEP 1 OF THE PIPELINE -- do this before anything else.
#
# The tokenizer decides how text gets chopped into the numbered pieces the model
# actually sees. Nothing else can happen until it exists, and it must then never
# change: a model trained against one tokenizer produces gibberish when read with
# another, because the ID for " the" is not the same number any more.
#
# "Training" here involves no gradients. It counts which adjacent pairs of
# characters occur most often and merges them, repeatedly, until the vocabulary
# holds 16,384 entries. See `src/minifrontier/tokenizer.py`.
#
# Output: a directory with `tokenizer.json` and `tokenizer_config.json`, the
# latter carrying a SHA-256 of the former so a mismatch is caught on load.

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from minifrontier.tokenizer import DIGIT_SPLIT_MODE, PRETOKENIZER_MODE, VOCAB_SIZE, train_byte_bpe


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
    parser.add_argument(
        "--digit-split",
        choices=("none", "no_leading_space", "leading_space"),
        default=DIGIT_SPLIT_MODE,
        help="digit pre-tokenization rule; the default is the only one used by real training",
    )
    parser.add_argument(
        "--pretokenizer",
        choices=("gpt2", "gpt4"),
        default=PRETOKENIZER_MODE,
        help=(
            "pre-tokenization regex family (MF-100); 'gpt2' is the only one used by "
            "real training so far -- 'gpt4' requires --digit-split none"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = train_byte_bpe(
        iter_input_text(args.inputs, jsonl_field=args.jsonl_field),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        digit_split=args.digit_split,
        pretokenizer=args.pretokenizer,
    )
    tokenizer.save(args.output)
    print(f"saved tokenizer with {tokenizer.vocab_size:,} entries to {args.output}")


if __name__ == "__main__":
    main()
