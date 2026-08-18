"""Export a native MiniFrontier release as a standalone Transformers/Hub repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.hf_export import export_transformers_repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_transformers_repository(
        args.release,
        args.output,
        source_revision=args.source_revision,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote load-tested Transformers repository to {args.output}")


if __name__ == "__main__":
    main()
