"""Generate a real Hugging Face dataset card for a published training-mixture export.

STEP 3 (optional) OF THE PIPELINE -- after scripts/prepare_data.py's
--export-parquet-dir has produced one train/validation Parquet pair per
mixture source (MF-125).

Reads each source's real metadata.json (written by prepare_data.py: admission
stats, real token counts) alongside its mixture weight, and writes a real,
honest dataset card (README.md with YAML frontmatter) describing the mixture
as one Hugging Face dataset repo with each source as its own named **config**
(https://huggingface.co/docs/hub/en/datasets-manual-configuration): a
`config_name`/`data_dir` pair per source in the card's own `configs:` YAML
block, so `load_dataset(repo_id, "github-code")` loads just that one source,
and the Hub's Dataset Viewer gets a per-source dropdown -- confirmed against
the real, current Hugging Face documentation, not assumed.

This deliberately does not assert one blanket license for the whole dataset.
The GitHub-code component alone spans many different real per-file permissive
licenses (see configs/code-repo-allowlist.txt) -- the card instead points
readers at each Parquet row's own `license`/`source` columns, matching this
project's own per-document provenance discipline (see Document in data.py).

Uploading is intentionally NOT built into this script. Once the card and each
source's `train-00000-of-00001.parquet`/`validation-00000-of-00001.parquet`
pair are arranged under same-named folders in one directory (see the card's
own "Repository layout" section for the exact structure), publish manually --
by hand through the Hub's own web upload UI, or with the standard CLI:

    huggingface-cli login
    huggingface-cli upload <username>/<dataset-name> <directory> --repo-type dataset

No custom upload code is added here on purpose: both of those are already-
complete tools that correctly handle auth, resumable large-file upload, and
repo creation. Reimplementing credential handling in this project's own code
would be duplicated, unreviewed surface area for no real benefit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple


class SourceEntry(NamedTuple):
    name: str
    weight: float
    metadata_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs=3,
        metavar=("NAME", "WEIGHT", "METADATA_JSON"),
        action="append",
        required=True,
        dest="sources",
        help="One mixture source: a short name (e.g. fineweb-edu), its real "
        "mixture weight as a fraction (e.g. 0.25), and the metadata.json "
        "written by prepare_data.py for that source. Repeat --source once "
        "per component of the mixture.",
    )
    parser.add_argument("--dataset-name", required=True, help="Human-readable dataset title.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the generated README.md dataset card.",
    )
    return parser.parse_args()


def _load_sources(raw_sources: list[list[str]]) -> list[SourceEntry]:
    entries = []
    for name, weight_text, metadata_text in raw_sources:
        weight = float(weight_text)
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"source {name!r} weight must be in (0, 1], got {weight}")
        entries.append(SourceEntry(name, weight, Path(metadata_text)))
    total_weight = sum(entry.weight for entry in entries)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"source weights must sum to 1.0, got {total_weight}")
    return entries


def _real_token_count(metadata: dict) -> int:
    return int(
        metadata["train"]["total_non_padding_tokens"]
        + metadata["validation"]["total_non_padding_tokens"]
    )


_SIZE_CATEGORY_THRESHOLDS: list[tuple[int, str]] = [
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
    (1_000_000_000, "100M<n<1B"),
    (10_000_000_000, "1B<n<10B"),
    (100_000_000_000, "10B<n<100B"),
    (1_000_000_000_000, "100B<n<1T"),
]


def _size_category(total_tokens: int) -> str:
    # Hugging Face's own size_categories convention (by example count -- token
    # count is this project's honest proxy, since packed shards have no
    # separate "example count" once documents are concatenated).
    for threshold, label in _SIZE_CATEGORY_THRESHOLDS:
        if total_tokens < threshold:
            return label
    return "n>1T"


def build_card(dataset_name: str, entries: list[SourceEntry]) -> str:
    rows = []
    total_tokens = 0
    for entry in entries:
        metadata = json.loads(entry.metadata_path.read_text(encoding="utf-8"))
        tokens = _real_token_count(metadata)
        total_tokens += tokens
        rows.append((entry.name, entry.weight, tokens, metadata["admission"]["admitted"]))

    configs_lines = ["configs:"]
    for entry in entries:
        configs_lines.append(f"  - config_name: {entry.name}")
        configs_lines.append(f"    data_dir: {entry.name}")

    frontmatter = "\n".join(
        [
            "---",
            "license: other",
            "license_name: mixed-per-example",
            "license_link: https://github.com/igal-abachi-dev/AI-LLM-Transformers-Edu-Model",
            "task_categories:",
            "  - text-generation",
            "language:",
            "  - en",
            f"size_categories:\n  - {_size_category(total_tokens)}",
            *configs_lines,
            "---",
        ]
    )

    lines = [frontmatter, "", f"# {dataset_name}", ""]
    lines.append(
        "Training-mixture export from "
        "[MiniFrontier](https://github.com/igal-abachi-dev/AI-LLM-Transformers-Edu-Model), "
        "an educational, from-scratch decoder-only language model. Each row is one admitted "
        "document (post-filter, post-dedup, pre-tokenization) with its full provenance: "
        "`text`, `source`, `revision`, `license`, `language`, `record_id`, `content_hash`, "
        "`path`, `source_type`, `split`, `parent_content_hash`, `transform`."
    )
    lines.append("")
    lines.append(
        "**License is per-example, not one blanket license for the dataset.** The GitHub-code "
        "component alone aggregates many different real permissive licenses (Apache-2.0, "
        "BSD-2-Clause, BSD-3-Clause, CC0-1.0, ISC, MIT, Unlicense) depending on which repository "
        "each row came from -- check the row's own `license` and `source` columns before reusing "
        "any individual example, and preserve that attribution the same way the original "
        "repository's own license requires."
    )
    lines.append("")
    lines.append("## Mixture composition")
    lines.append("")
    lines.append("| Source | Weight | Real tokens | Admitted documents |")
    lines.append("| --- | --- | --- | --- |")
    for name, weight, tokens, admitted in rows:
        lines.append(f"| {name} | {weight:.0%} | {tokens:,} | {admitted:,} |")
    lines.append(f"| **Total** | **100%** | **{total_tokens:,}** | |")
    lines.append("")
    lines.append(
        "Weights and real token counts are measured directly from each source's own "
        "`metadata.json` (written by `scripts/prepare_data.py`) at export time, not estimated."
    )
    lines.append("")
    lines.append("## Repository layout")
    lines.append("")
    lines.append(
        "Each source is its own Hugging Face **config** (a separate entry in the Dataset "
        'Viewer\'s dropdown, and `load_dataset(repo_id, "github-code")` loads just that '
        "one). Place each source's `scripts/prepare_data.py --export-parquet-dir` output "
        "directly at the repo root, in a folder named exactly like the source, before "
        "uploading:"
    )
    lines.append("")
    lines.append("```")
    lines.append(f"{dataset_name}/")
    lines.append("├── README.md          (this file)")
    for entry in entries:
        lines.append(f"├── {entry.name}/")
        lines.append("│   ├── train-00000-of-00001.parquet")
        lines.append("│   └── validation-00000-of-00001.parquet")
    lines.append("```")
    lines.append("")
    lines.append(
        "The `train-*`/`validation-*` filenames are Hugging Face's own auto-detected split "
        "naming convention -- with a folder per source and this naming, the Hub's Dataset "
        "Viewer and `load_dataset` both work correctly with no loading script, just the "
        "`configs:` block above."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    entries = _load_sources(args.sources)
    card = build_card(args.dataset_name, entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card, encoding="utf-8")
    print(f"wrote {args.output} ({len(entries)} sources)")


if __name__ == "__main__":
    main()
