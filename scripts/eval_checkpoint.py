"""Evaluate a raw train/pretrain.py checkpoint against held-out packed shards.

Fills a real gap: scripts/eval.py only accepts a full export_release directory
(minifrontier.checkpoint.load_release), but every bounded-ablation comparison
this project runs (MF-081/082/107/108) produces raw train/pretrain.py
checkpoints, never a full release export -- a throwaway comparison arm doesn't
warrant one. batches_from_packed_shards/evaluate_token_batches is the standing
recipe (MF-086) every other bounded comparison already reuses; this script is
the one place that recipe runs on demand against an arbitrary already-trained
raw checkpoint, without retraining or exporting anything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from minifrontier.checkpoint import load_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.evaluation.validation import batches_from_packed_shards, evaluate_token_batches
from minifrontier.model import MiniFrontier
from minifrontier.shards import PackedShardDataset
from minifrontier.tokenizer import MiniFrontierTokenizer

VALIDATION_BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="directory containing config.json"
    )
    parser.add_argument("--validation-shards", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--batch-size", type=int, default=VALIDATION_BATCH_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint: Path,
    validation_shards: Path,
    tokenizer_dir: Path,
    *,
    batch_size: int = VALIDATION_BATCH_SIZE,
    device: str = "cpu",
) -> dict[str, object]:
    config = ModelConfig(**json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    model = MiniFrontier(config).to(device)
    load_training_checkpoint(checkpoint, model, trusted_local_state=True)
    tokenizer = MiniFrontierTokenizer.from_directory(tokenizer_dir)
    dataset = PackedShardDataset(validation_shards)
    metrics = evaluate_token_batches(
        model,
        batches_from_packed_shards(dataset, tokenizer, batch_size=batch_size, device=device),
        pad_id=tokenizer.pad_id,
    )
    return {"checkpoint": str(checkpoint), **asdict(metrics)}


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        args.validation_shards,
        args.tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
