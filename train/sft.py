"""Assistant-only raw-PyTorch supervised fine-tuning entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from minifrontier.checkpoint import load_training_checkpoint, save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.reproducibility import seed_everything
from minifrontier.sft import encode_sft_example, iter_conversations, pack_sft_examples
from minifrontier.tokenizer import MiniFrontierTokenizer
from minifrontier.training import (
    ShuffledBatchProvider,
    TrainingBatch,
    TrainingConfig,
    TrainingState,
    WarmupCosineSchedule,
    build_adamw,
    train_updates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_packs(packs: list[TrainingBatch], batch_size: int) -> list[TrainingBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches = []
    for start in range(0, len(packs), batch_size):
        group = packs[start : start + batch_size]
        if len(group) < batch_size:
            break
        batches.append(
            TrainingBatch(
                torch.cat([batch.tokens for batch in group]),
                loss_mask=torch.cat(
                    [batch.loss_mask for batch in group if batch.loss_mask is not None]
                ),
            )
        )
    if not batches:
        raise ValueError("SFT dataset does not contain one complete configured batch")
    return batches


def main() -> None:
    args = parse_args()
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    seed_everything(args.seed, deterministic=args.device == "cpu")
    config = ModelConfig(
        **json.loads((args.base_checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    model = MiniFrontier(config)
    load_training_checkpoint(
        args.base_checkpoint,
        model,
        restore_rng=False,
        trusted_local_state=True,
    )
    tokenizer = MiniFrontierTokenizer.from_directory(args.tokenizer)
    records = list(iter_conversations(args.dataset))
    examples = [
        encode_sft_example(record, tokenizer, max_length=config.max_seq_len) for record in records
    ]
    packs = list(
        pack_sft_examples(
            examples,
            sequence_length=config.max_seq_len,
            pad_id=tokenizer.pad_id,
        )
    )
    provider = ShuffledBatchProvider(
        _batch_packs(packs, args.batch_size),
        seed=args.seed,
    )
    training = TrainingConfig(
        max_updates=args.updates,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_updates=args.warmup_updates,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        precision=args.precision,
        attention_impl="sdpa" if args.device == "cpu" else None,
    )
    optimizer = build_adamw(model, training)[0]
    schedule = WarmupCosineSchedule(training)
    state = TrainingState()
    lineage = {
        "base_weights_sha256": _sha256(args.base_checkpoint / "model.safetensors"),
        "dataset_sha256": _sha256(args.dataset),
        "records": len(records),
    }
    if args.resume is not None:
        trainer, cursor = load_training_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=schedule,
            trusted_local_state=True,
        )
        if trainer.get("lineage") != lineage or trainer.get("training_config") != asdict(training):
            raise ValueError("SFT resume lineage/training configuration does not match")
        state = TrainingState.from_dict(trainer["training_state"])
        provider.load_state_dict(cursor)
    args.output.mkdir(parents=True, exist_ok=True)

    def checkpoint_callback(
        current_model,
        current_optimizer,
        current_schedule,
        current_state,
    ) -> None:
        if current_state.completed_updates % args.checkpoint_interval:
            return
        save_training_checkpoint(
            args.output / f"checkpoint-{current_state.completed_updates:08d}",
            current_model,
            optimizer=current_optimizer,
            scheduler=current_schedule,
            trainer_state={
                "stage": "assistant_only_sft",
                "training_state": current_state.to_dict(),
                "training_config": asdict(training),
                "lineage": lineage,
            },
            data_cursor=provider.state_dict(),
        )

    optimizer, schedule, state, _ = train_updates(
        model,
        provider,
        training,
        device=args.device,
        optimizer=optimizer,
        schedule=schedule,
        state=state,
        update_callback=checkpoint_callback,
    )
    save_training_checkpoint(
        args.output / "final",
        model,
        optimizer=optimizer,
        scheduler=schedule,
        trainer_state={
            "stage": "assistant_only_sft",
            "training_state": state.to_dict(),
            "training_config": asdict(training),
            "lineage": lineage,
        },
        data_cursor=provider.state_dict(),
    )
    print(f"completed {state.completed_updates} assistant-only SFT updates")


if __name__ == "__main__":
    main()
