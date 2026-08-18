"""Convert a Transformers export using a pinned llama.cpp checkout with real MF support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.gguf import convert_high_precision_gguf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--llama-cpp-revision", required=True)
    parser.add_argument("--transformers-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_high_precision_gguf(
        llama_cpp=args.llama_cpp,
        llama_cpp_revision=args.llama_cpp_revision,
        transformers_repository=args.transformers_repository,
        output=args.output,
        outtype=args.outtype,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote high-precision MiniFrontier GGUF to {args.output}")


if __name__ == "__main__":
    main()
