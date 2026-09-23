"""Build deduplicated, contamination-checked, immutable MiniFrontier token shards."""

# STEP 2 OF THE PIPELINE -- needs a tokenizer from step 1.
#
# Turns raw streamed text into the fixed-length integer sequences training reads.
# In order: stream documents, drop junk and duplicates, split train/validation by
# content hash, tokenize, pack end to end, and write memory-mapped shards.
#
# Why do this ahead of time instead of during training? Speed, and repeatability.
# The shards are hashed and immutable, so a run can state exactly which bytes it
# trained on, and an interrupted run can resume on the next unseen row rather than
# re-reading data the model has already seen.
#
# "Contamination-checked" means validation and benchmark documents are excluded
# from training by hash. Skip that and your evaluation scores measure memorization
# instead of learning. See `src/minifrontier/data.py` and `shards.py`.

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from minifrontier.code_data import CodeAdmissionStats, filter_code_documents
from minifrontier.data import (
    iter_cosmopedia_v2,
    iter_dclm_edu,
    iter_ebook_markdown,
    iter_finemath,
    iter_fineweb_edu,
    iter_github_code,
    iter_github_code_from_repos,
    iter_jsonl_documents,
    split_bucket,
)
from minifrontier.shards import (
    AdmissionStats,
    DiskDeduplicator,
    ParquetDocumentWriter,
    TokenShardWriter,
    admit_documents,
)
from minifrontier.tokenizer import MiniFrontierTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument(
        "--source",
        choices=(
            "fineweb-edu",
            "dclm-edu",
            "finemath",
            "cosmopedia-v2",
            "github-code",
            "ebook-markdown",
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument(
        "--dclm-min-score",
        type=int,
        default=3,
        help="Only meaningful with --source dclm-edu: minimum edu_int_score to admit "
        "(the dataset's own real range is 2-5). 3 is the MF-094/MF-095 mixture "
        "proposal's own cutoff.",
    )
    parser.add_argument(
        "--finemath-config",
        default="finemath-4plus",
        choices=("finemath-3plus", "finemath-4plus", "infiwebmath-3plus", "infiwebmath-4plus"),
        help="Only meaningful with --source finemath: which of the dataset's four real "
        "subsets to use.",
    )
    parser.add_argument(
        "--github-languages",
        nargs="*",
        help="Only meaningful with --source github-code: restrict to these languages "
        "(case-insensitive, e.g. Python JavaScript). Omit for no language restriction.",
    )
    parser.add_argument(
        "--github-repo-allowlist",
        type=Path,
        help="Only meaningful with --source github-code: a text file, one 'owner/repo' "
        "per line, restricting the stream to exactly this curated set of repositories "
        "instead of an unfiltered crawl of the whole dataset. When given, each repo is "
        "fetched directly and currently (MF-134: a real, persistent, incrementally-"
        "updated bare-mirror cache under data/github-code-cache/, not the static "
        "codeparrot/github-code dataset snapshot).",
    )
    parser.add_argument(
        "--github-force-refresh",
        action="store_true",
        help="Only meaningful with --source github-code --github-repo-allowlist: "
        "bypass the mirror cache's staleness check and re-clone every allowlisted "
        "repo from scratch, regardless of how recently it was last fetched.",
    )
    parser.add_argument(
        "--ebook-directory",
        type=Path,
        help="Only meaningful with --source ebook-markdown: a directory already "
        "processed by the standalone pdf-to-markdown-rag pipeline, read at "
        "<directory>/md/<book-id>/book.md per book (MF-124).",
    )
    parser.add_argument(
        "--ebook-license",
        default="Public Domain",
        help="Only meaningful with --source ebook-markdown: the license string every "
        "book in --ebook-directory is asserted to carry (MF-124: real ingestion is "
        "public-domain-only; this is not detected from the pipeline's own per-book "
        "license.json, which is deliberately not read).",
    )
    parser.add_argument(
        "--ebook-revision",
        default="n/a",
        help="Only meaningful with --source ebook-markdown: ebooks have no natural "
        "per-document revision concept (no repository commit or dataset snapshot); "
        "defaults to a static placeholder.",
    )
    parser.add_argument(
        "--ebook-language",
        default="English",
        help="Only meaningful with --source ebook-markdown: applies uniformly to "
        "every book in --ebook-directory -- run once per language for a mixed-"
        "language batch.",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help=(
            "If a previous interrupted invocation left --output or "
            "--export-parquet-dir populated without a completed metadata.json, "
            "delete both partial outputs and restart preprocessing from the "
            "beginning. Completed output is never deleted."
        ),
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--sequences-per-shard", type=int, default=1024)
    parser.add_argument("--evaluation-signatures", type=Path)
    parser.add_argument("--keep-remainder", action="store_true")
    parser.add_argument(
        "--packing",
        choices=("ribbon", "best_fit", "bos_crop"),
        default="ribbon",
        help=(
            "MF-097: ribbon (default, unchanged) concatenates documents and slices "
            "fixed-length rows immediately. best_fit/bos_crop buffer whole documents "
            "and pack a batch at a time -- see packing.py for the real tradeoff."
        ),
    )
    parser.add_argument(
        "--pack-buffer-documents",
        type=int,
        default=256,
        help="Only meaningful with --packing best_fit/bos_crop: how many whole "
        "documents to buffer before packing a batch.",
    )
    parser.add_argument(
        "--export-parquet-dir",
        type=Path,
        help="MF-125: optionally also snapshot the admitted, pre-tokenization "
        "Document rows (raw text plus every provenance field) to this directory, "
        "as train-00000-of-00001.parquet / validation-00000-of-00001.parquet -- "
        "the naming Hugging Face's datasets library auto-detects as split files, "
        "so this directory can be uploaded directly as one config's data_dir in a "
        "published dataset repo (see scripts/build_dataset_card.py). A "
        "publish-oriented export, not a training input. Omit for the existing "
        "shards-only behavior.",
    )
    return parser.parse_args()


def prepare_output_directories(
    output: Path,
    export_parquet_dir: Path | None,
    *,
    restart_incomplete: bool,
) -> None:
    """Reject existing publication output, or explicitly clear an incomplete run.

    Preprocessing has no document-level resume. ``metadata.json`` is atomically
    published only after both shard and optional Parquet finalization, so its
    absence is the fail-closed signal that every existing output is disposable.
    """

    paths = [output]
    if export_parquet_dir is not None:
        paths.append(export_parquet_dir)

    populated: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if not path.is_dir():
            raise FileExistsError(f"publication path exists but is not a directory: {path}")
        if any(path.iterdir()):
            populated.append(path)

    if not populated:
        return
    if not restart_incomplete:
        joined = ", ".join(str(path) for path in populated)
        raise FileExistsError(
            "publication output must be absent or empty; remove incomplete output "
            f"or pass --restart-incomplete: {joined}"
        )

    completion_marker = output / "metadata.json"
    if completion_marker.exists():
        raise FileExistsError(
            f"refusing to delete completed preprocessing output: {completion_marker} exists"
        )

    for path in populated:
        print(f"Removing incomplete preprocessing output before restart: {path}")
        shutil.rmtree(path)


def document_stream(args: argparse.Namespace, *, code_stats: CodeAdmissionStats | None = None):
    """Select a bounded streaming source without creating an intermediate corpus file.

    ``code_stats``, when given, is only consulted for ``--source github-code``:
    that branch routes through `filter_code_documents` (secret/personal-data
    redaction, generated/vendor-path rejection -- see `code_data.py`) before
    `admit_documents`'s own generic length/dedup/contamination checks ever see
    it. Every other source is prose, not code, and is unaffected.
    """

    if args.manifest is not None:
        if args.limit is not None or args.start or args.shuffle_seed is not None:
            raise ValueError("cursor/shuffle options require a streaming --source")
        return iter_jsonl_documents(args.manifest)
    if args.source == "fineweb-edu":
        return iter_fineweb_edu(
            limit=args.limit,
            start=args.start,
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    if args.source == "dclm-edu":
        return iter_dclm_edu(
            min_edu_int_score=args.dclm_min_score,
            limit=args.limit,
            start=args.start,
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    if args.source == "finemath":
        return iter_finemath(
            config=args.finemath_config,
            limit=args.limit,
            start=args.start,
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    if args.source == "cosmopedia-v2":
        return iter_cosmopedia_v2(
            limit=args.limit,
            start=args.start,
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    if args.source == "github-code":
        code_stats = code_stats if code_stats is not None else CodeAdmissionStats()
        if args.github_repo_allowlist is not None:
            repo_names = [
                line.strip()
                for line in args.github_repo_allowlist.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            # MF-070 (2026-09-15): a real allowlist means a real, specific
            # target list -- fetch each repo directly and currently (real
            # `git clone --depth 1`) instead of scanning
            # codeparrot/github-code, a static 2022-03-16 snapshot, for
            # them. See iter_github_code_from_repos's own docstring.
            stream = iter_github_code_from_repos(
                repo_names,
                languages=args.github_languages,
                limit=args.limit,
                start=args.start,
                shuffle_seed=args.shuffle_seed,
                force_refresh=args.github_force_refresh,
            )
        else:
            stream = iter_github_code(
                languages=args.github_languages,
                repo_names=None,
                limit=args.limit,
                start=args.start,
                shuffle_seed=args.shuffle_seed,
                shuffle_buffer=args.shuffle_buffer,
            )
        return filter_code_documents(stream, stats=code_stats)
    if args.source == "ebook-markdown":
        if args.ebook_directory is None:
            raise ValueError("--source ebook-markdown requires --ebook-directory")
        return iter_ebook_markdown(
            args.ebook_directory,
            license=args.ebook_license,
            revision=args.ebook_revision,
            language=args.ebook_language,
            limit=args.limit,
            start=args.start,
            shuffle_seed=args.shuffle_seed,
        )
    raise ValueError(f"unsupported data source: {args.source}")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    prepare_output_directories(
        args.output,
        args.export_parquet_dir,
        restart_incomplete=args.restart_incomplete,
    )
    tokenizer = MiniFrontierTokenizer.from_directory(args.tokenizer)
    args.output.mkdir(parents=True, exist_ok=True)
    signatures = {"exact": [], "simhash": []}
    if args.evaluation_signatures is not None:
        signatures = json.loads(args.evaluation_signatures.read_text(encoding="utf-8"))
    stats = AdmissionStats()
    code_stats = CodeAdmissionStats()
    train_writer = TokenShardWriter(
        args.output / "train",
        tokenizer,
        sequence_length=args.sequence_length,
        sequences_per_shard=args.sequences_per_shard,
        packing=args.packing,
        pack_buffer_documents=args.pack_buffer_documents,
    )
    validation_writer = TokenShardWriter(
        args.output / "validation",
        tokenizer,
        sequence_length=args.sequence_length,
        sequences_per_shard=args.sequences_per_shard,
        packing=args.packing,
        pack_buffer_documents=args.pack_buffer_documents,
    )
    validation_threshold = int(args.validation_fraction * 10_000)
    parquet_train_writer = None
    parquet_validation_writer = None
    if args.export_parquet_dir is not None:
        args.export_parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_train_writer = ParquetDocumentWriter(
            args.export_parquet_dir / "train-00000-of-00001.parquet"
        )
        parquet_validation_writer = ParquetDocumentWriter(
            args.export_parquet_dir / "validation-00000-of-00001.parquet"
        )
    with DiskDeduplicator(
        args.output / "dedup-signatures.sqlite",
        max_hamming_distance=stats.max_hamming_distance,
    ) as deduplicator:
        admitted = admit_documents(
            document_stream(args, code_stats=code_stats),
            deduplicator,
            stats=stats,
            evaluation_exact_hashes=set(signatures["exact"]),
            evaluation_simhashes={int(value, 16) for value in signatures["simhash"]},
        )
        for document in admitted:
            bucket = split_bucket(document)
            is_validation = bucket < validation_threshold
            writer = validation_writer if is_validation else train_writer
            writer.add(document)
            if parquet_train_writer is not None:
                active_parquet_writer = (
                    parquet_validation_writer if is_validation else parquet_train_writer
                )
                active_parquet_writer.add(document)
    train_manifest = train_writer.finalize(drop_remainder=not args.keep_remainder)
    validation_manifest = validation_writer.finalize(drop_remainder=not args.keep_remainder)
    exported_parquet_train_rows = (
        parquet_train_writer.finalize() if parquet_train_writer is not None else None
    )
    exported_parquet_validation_rows = (
        parquet_validation_writer.finalize() if parquet_validation_writer is not None else None
    )
    metadata = {
        "admission": asdict(stats),
        "code_admission": asdict(code_stats) if args.source == "github-code" else None,
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
        "export_parquet_dir": (
            str(args.export_parquet_dir) if args.export_parquet_dir is not None else None
        ),
        "export_parquet_train_rows": exported_parquet_train_rows,
        "export_parquet_validation_rows": exported_parquet_validation_rows,
    }
    metadata_path = args.output / "metadata.json"
    temporary_metadata_path = args.output / ".metadata.tmp"
    temporary_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata_path, metadata_path)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
