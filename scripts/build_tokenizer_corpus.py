"""Build a raw-text JSONL corpus for tokenizer training (step 0, before step 1).

Not the token shards `prepare_data.py` builds -- this has no tokenizer input and
produces no split/dedup/pack pipeline. It exists purely to feed
`scripts/train_tokenizer.py`'s own `--jsonl-field text` input: a bounded sample of
real documents, drawn from the same real, provenance-approved sources as model
training, at ratios chosen for what a *tokenizer* needs (enough real exposure to
each domain's distinct structural patterns to learn efficient merges) rather than
what the model's own training mixture is optimizing for (downstream capability).

See MF-119 in tasks/backlog.md: the previous tokenizer corpus (14.7MB, 3,000
documents) was verified to be real-code-light -- only 6.4% of documents contained
any code-shaped content, so multi-space indentation never appeared often enough
for BPE to learn to compress it (a 7-space indent tokenized as 7 separate
single-space tokens). The default ratios below (DCLM-Edu 45% / FineWeb-Edu 30% /
GitHub-code 15% / FineMath 5% / Cosmopedia-v2 5%) are `wsd-fixed`'s real mixture,
not `general-purpose-optimized`'s (the model's own final training-mixture choice,
MF-095) -- deliberately: the tokenizer's own corpus doesn't need to track whichever
ratio wins the model's downstream-capability question, it only needs a real,
non-trivial code share, and 15% was independently judged enough for that here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minifrontier.data import (
    Document,
    iter_cosmopedia_v2,
    iter_dclm_edu,
    iter_finemath,
    iter_fineweb_edu,
    iter_github_code,
)

_DEFAULT_RATIOS: dict[str, float] = {
    "dclm-edu": 0.45,
    "fineweb-edu": 0.30,
    "github-code": 0.15,
    "finemath": 0.05,
    "cosmopedia-v2": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--total-documents",
        type=int,
        default=6_000,
        help="Split across sources by --*-ratio below and rounded per source. "
        "6,000 (roughly double the previous 3,000-document/14.7MB corpus) is a "
        "modest increase proportionate to this project's own small (16,384) "
        "vocabulary target, not a jump to production-tokenizer-corpus scale.",
    )
    parser.add_argument("--dclm-ratio", type=float, default=_DEFAULT_RATIOS["dclm-edu"])
    parser.add_argument("--fineweb-ratio", type=float, default=_DEFAULT_RATIOS["fineweb-edu"])
    parser.add_argument("--github-ratio", type=float, default=_DEFAULT_RATIOS["github-code"])
    parser.add_argument("--finemath-ratio", type=float, default=_DEFAULT_RATIOS["finemath"])
    parser.add_argument("--cosmopedia-ratio", type=float, default=_DEFAULT_RATIOS["cosmopedia-v2"])
    parser.add_argument("--dclm-min-score", type=int, default=3)
    parser.add_argument("--finemath-config", default="finemath-4plus")
    parser.add_argument(
        "--github-repo-allowlist",
        type=Path,
        default=Path("configs/code-repo-allowlist.txt"),
        help="Same curated, provenance-approved allowlist real training shards use.",
    )
    parser.add_argument(
        "--github-languages",
        nargs="*",
        default=None,
        help="Omit (the default) for no language restriction -- the tokenizer "
        "benefits from the allowlist's whole real language mix, not one language.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    return parser.parse_args()


def _read_repo_allowlist(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main() -> None:
    args = parse_args()
    ratios = {
        "dclm-edu": args.dclm_ratio,
        "fineweb-edu": args.fineweb_ratio,
        "github-code": args.github_ratio,
        "finemath": args.finemath_ratio,
        "cosmopedia-v2": args.cosmopedia_ratio,
    }
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total_ratio}")

    repo_names = _read_repo_allowlist(args.github_repo_allowlist)
    limits = {name: round(args.total_documents * ratio) for name, ratio in ratios.items()}
    streams: dict[str, object] = {
        "dclm-edu": iter_dclm_edu(
            min_edu_int_score=args.dclm_min_score,
            limit=limits["dclm-edu"],
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        ),
        "fineweb-edu": iter_fineweb_edu(
            limit=limits["fineweb-edu"],
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        ),
        "github-code": iter_github_code(
            languages=args.github_languages,
            repo_names=repo_names,
            limit=limits["github-code"],
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        ),
        "finemath": iter_finemath(
            config=args.finemath_config,
            limit=limits["finemath"],
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        ),
        "cosmopedia-v2": iter_cosmopedia_v2(
            limit=limits["cosmopedia-v2"],
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer=args.shuffle_buffer,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    total_bytes = 0
    with args.output.open("w", encoding="utf-8") as file:
        for name, documents in streams.items():
            written = 0
            for document in documents:
                assert isinstance(document, Document)
                line = document.to_json()
                file.write(line + "\n")
                total_bytes += len(line.encode("utf-8")) + 1
                written += 1
            counts[name] = written
            print(f"{name}: wanted {limits[name]}, wrote {written}")

    total_written = sum(counts.values())
    print(f"total documents: {total_written}, total bytes: {total_bytes}")
    for name, wanted in limits.items():
        if counts[name] < wanted:
            print(
                f"warning: {name} stream ran out early ({counts[name]}/{wanted}) -- "
                "the real source or allowlist may be smaller than requested"
            )


if __name__ == "__main__":
    main()
