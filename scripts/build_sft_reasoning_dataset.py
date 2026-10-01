"""Build MF-123's real, provenance-tracked step-by-step reasoning SFT dataset.

Beginner's map of this file
----------------------------
The only SFT data this project had until now was a 32-example placeholder of
plain instruction-following (`"Reply with exactly the word RED"` -> `"RED"`) --
no worked reasoning at all. `MF-116`'s decision was that the real SFT stage
needs demonstrations that show intermediate steps, not just a final answer.

This script sources those demonstrations from GSM8K's own real training split
(`openai/gsm8k`, config `main`) -- distinct from its test split, which this
project's own eval harness (`scripts/eval.py --include-gsm8k`) already scores
against (`lm_eval`'s installed `gsm8k.yaml`: `training_split: train`,
`test_split: test`). Every GSM8K answer already *is* a real, worked,
multi-step solution (including inline `<<calculation=result>>` annotations)
ending in `#### <final answer>` -- used here verbatim, not rewritten or
synthesized, matching this project's "real data, real provenance" discipline.

Runs fully offline against the already-cached dataset (`HF_HUB_OFFLINE=1`) --
no network access needed on a machine that has already run
`scripts/eval.py --include-gsm8k` once before, which is what populated the
local cache this reads from.

Run it with::

    HF_HUB_OFFLINE=1 .venv\\Scripts\\python.exe scripts/build_sft_reasoning_dataset.py \\
        --output data/sft/gsm8k-reasoning-v1.jsonl

``--limit`` caps how many of the real 7,473 train examples are emitted --
deliberately left undecided by this script (full set by default): how much of
it to actually *train* on is a decision for whoever runs `train/sft.py`, not
baked into the data-prep step.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import load_dataset

from minifrontier.chat import ChatMessage
from minifrontier.sft import ConversationRecord, conversation_hash

SOURCE_DATASET = "openai/gsm8k"
SOURCE_CONFIG = "main"
# The exact HF dataset-repo revision this project's local cache was built
# from (`hf://datasets/openai/gsm8k@<this>/main/train-00000-of-00001.parquet`,
# read directly out of the cache's own `dataset_info.json`), not an assumed
# "latest" -- so this provenance record stays correct even if the Hub's own
# `main` ref moves later.
SOURCE_REVISION = "740312add88f781978c0658806c59bc2815b9866"
# GSM8K's license per OpenAI's own `openai/grade-school-math` GitHub
# repository (MIT) -- well-established, widely-cited public documentation of
# this specific dataset's terms; not independently re-fetched from GitHub in
# this session, disclosed rather than silently assumed.
SOURCE_LICENSE = "MIT"


def _normalized_question(question: str) -> str:
    return " ".join(question.split()).strip().lower()


def build_records(limit: int | None) -> list[ConversationRecord]:
    train = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train")
    records = []
    for index, row in enumerate(train):
        if limit is not None and index >= limit:
            break
        messages = (
            ChatMessage(role="user", content=str(row["question"])),
            ChatMessage(role="assistant", content=str(row["answer"])),
        )
        records.append(
            ConversationRecord(
                messages=messages,
                source=SOURCE_DATASET,
                revision=SOURCE_REVISION,
                license=SOURCE_LICENSE,
                record_id=f"gsm8k-train-{index:05d}",
                content_hash=conversation_hash(messages),
            )
        )
    return records


def check_no_train_test_overlap() -> int:
    """Real contamination check (MF-123's own explicit requirement): confirm
    GSM8K's train and test splits do not share a question, rather than
    trusting the split names alone. Returns the real overlap count found."""

    train = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train")
    test = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="test")
    train_questions = {_normalized_question(str(row["question"])) for row in train}
    test_questions = {_normalized_question(str(row["question"])) for row in test}
    return len(train_questions & test_questions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlap = check_no_train_test_overlap()
    if overlap:
        raise ValueError(
            f"real contamination found: {overlap} question(s) appear in both GSM8K's "
            "train and test splits -- refusing to build the SFT dataset until resolved"
        )
    records = build_records(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            # `asdict` recurses into the nested `ChatMessage` dataclasses too,
            # so `messages` already comes out as plain {"role", "content"}
            # dicts -- exactly `ConversationRecord.from_mapping`'s expected shape.
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    print(
        f"wrote {len(records)} real, provenance-tracked reasoning examples to "
        f"{args.output} (source={SOURCE_DATASET}@{SOURCE_REVISION}, "
        f"license={SOURCE_LICENSE}, train/test overlap={overlap})"
    )


if __name__ == "__main__":
    main()
