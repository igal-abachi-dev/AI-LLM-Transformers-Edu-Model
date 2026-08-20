"""Validate and sanitize a provenance-complete permissive code manifest."""

# The gate that code training data has to pass. Every record must name its
# repository, revision, license, path and content hash, and the license must be on
# the permissive list. Anything missing is rejected rather than assumed -- an
# unlabelled file is not "probably fine", it is unusable.
#
# The same pass screens for secrets, because public repositories genuinely do
# contain leaked API keys, and a model that memorizes one will reproduce it on
# request.
#
# See `src/minifrontier/code_data.py` and `docs/DATA_GOVERNANCE.md`.

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from minifrontier.code_data import (
    CodeAdmissionStats,
    CodeFilterConfig,
    filter_code_documents,
)
from minifrontier.data import iter_jsonl_documents
from minifrontier.shards import AdmissionStats, DiskDeduplicator, admit_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-signatures", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("output directory must be absent or empty for immutable publication")
    args.output.mkdir(parents=True, exist_ok=True)
    signatures = {"exact": [], "simhash": []}
    if args.evaluation_signatures is not None:
        signatures = json.loads(args.evaluation_signatures.read_text(encoding="utf-8"))
    code_stats = CodeAdmissionStats()
    dedup_stats = AdmissionStats()
    temporary = args.output / ".approved-code.jsonl.tmp"
    final = args.output / "approved-code.jsonl"
    with DiskDeduplicator(args.output / "code-signatures.sqlite") as deduplicator:
        structurally_approved = filter_code_documents(
            iter_jsonl_documents(args.manifest),
            config=CodeFilterConfig(),
            stats=code_stats,
        )
        admitted = admit_documents(
            structurally_approved,
            deduplicator,
            stats=dedup_stats,
            evaluation_exact_hashes=set(signatures["exact"]),
            evaluation_simhashes={int(value, 16) for value in signatures["simhash"]},
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for document in admitted:
                file.write(document.to_json() + "\n")
    os.replace(temporary, final)
    report = {"code_admission": asdict(code_stats), "deduplication": asdict(dedup_stats)}
    (args.output / "admission-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
