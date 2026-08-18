"""Build deduplicated, contamination-checked, immutable MiniFrontier token shards."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from minifrontier.data import iter_fineweb_edu, iter_jsonl_documents, split_bucket
from minifrontier.shards import (
    AdmissionStats,
    DiskDeduplicator,
    TokenShardWriter,
    admit_documents,
)
from minifrontier.tokenizer import MiniFrontierTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--source", choices=("fineweb-edu",))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--sequences-per-shard", type=int, default=1024)
    parser.add_argument("--evaluation-signatures", type=Path)
    parser.add_argument("--keep-remainder", action="store_true")
    return parser.parse_args()


def document_stream(args: argparse.Namespace):
    """Select a bounded streaming source without creating an intermediate corpus file."""

    if args.manifest is not None:
        if args.limit is not None or args.start or args.shuffle_seed is not None:
            raise ValueError("FineWeb cursor/shuffle options require --source fineweb-edu")
        return iter_jsonl_documents(args.manifest)
    if args.source == "fineweb-edu":
        return iter_fineweb_edu(
            limit=args.limit,
            start=args.start,
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    raise ValueError(f"unsupported data source: {args.source}")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("output directory must be absent or empty for immutable publication")
    tokenizer = MiniFrontierTokenizer.from_directory(args.tokenizer)
    args.output.mkdir(parents=True, exist_ok=True)
    signatures = {"exact": [], "simhash": []}
    if args.evaluation_signatures is not None:
        signatures = json.loads(args.evaluation_signatures.read_text(encoding="utf-8"))
    stats = AdmissionStats()
    train_writer = TokenShardWriter(
        args.output / "train",
        tokenizer,
        sequence_length=args.sequence_length,
        sequences_per_shard=args.sequences_per_shard,
    )
    validation_writer = TokenShardWriter(
        args.output / "validation",
        tokenizer,
        sequence_length=args.sequence_length,
        sequences_per_shard=args.sequences_per_shard,
    )
    validation_threshold = int(args.validation_fraction * 10_000)
    with DiskDeduplicator(
        args.output / "dedup-signatures.sqlite",
        max_hamming_distance=stats.max_hamming_distance,
    ) as deduplicator:
        admitted = admit_documents(
            document_stream(args),
            deduplicator,
            stats=stats,
            evaluation_exact_hashes=set(signatures["exact"]),
            evaluation_simhashes={int(value, 16) for value in signatures["simhash"]},
        )
        for document in admitted:
            bucket = split_bucket(document)
            writer = validation_writer if bucket < validation_threshold else train_writer
            writer.add(document)
    train_manifest = train_writer.finalize(drop_remainder=not args.keep_remainder)
    validation_manifest = validation_writer.finalize(drop_remainder=not args.keep_remainder)
    metadata = {
        "admission": asdict(stats),
        "source": args.source or "manifest",
        "source_manifest": str(args.manifest) if args.manifest is not None else None,
        "source_start": args.start if args.source is not None else None,
        "source_limit": args.limit if args.source is not None else None,
        "source_shuffle_seed": args.shuffle_seed if args.source is not None else None,
        "source_shuffle_buffer": args.shuffle_buffer if args.source is not None else None,
        "sequence_length": args.sequence_length,
        "validation_fraction": args.validation_fraction,
        "train": asdict(train_manifest),
        "validation": asdict(validation_manifest),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
