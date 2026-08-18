"""Apply a deterministic, versioned FIM mixture to an approved code manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from minifrontier.code_data import FIM_TRANSFORM_VERSION, mix_fim_documents
from minifrontier.data import iter_jsonl_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    total = 0
    transformed = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        originals = iter_jsonl_documents(args.manifest)
        for document in mix_fim_documents(originals, rate=args.rate, seed=args.seed):
            total += 1
            transformed += int(":fim:" in document.record_id)
            file.write(document.to_json() + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "version": FIM_TRANSFORM_VERSION,
                "rate": args.rate,
                "seed": args.seed,
                "documents": total,
                "transformed": transformed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
