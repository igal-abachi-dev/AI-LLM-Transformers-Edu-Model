"""Canonical single-process MiniFrontier pretraining entry point."""

# STEP 3 OF THE PIPELINE, and the one that takes the hours. This is where a
# randomly initialized model becomes one that has read a lot of text.
#
# What happens here, at a glance: build the model from a config, wire up the data
# shards from step 2, and hand both to `train_updates` in
# `src/minifrontier/training.py` -- which is the actual loop and the file to read
# if you want to understand training itself.
#
# A few things worth knowing before your first run:
#
# * A "step" means one optimizer update, not one batch. With gradient accumulation
#   several batches contribute to a single update.
# * The run is resumable. It checkpoints weights, optimizer state, schedule
#   position and the data cursor together, so an interrupted run continues exactly
#   where it stopped rather than re-reading data it has already learned from.
# * Loss is reported in nats per token. Starting value is around ln(vocab_size),
#   about 9.7 for a 16,384-token vocabulary -- that is the model guessing blind.
# * Do not start a serious run until the tests, the overfit proof, and the data
#   checks pass. Debugging at hour six is far more expensive than at minute one.

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from minifrontier.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from minifrontier.compilation import maybe_compile
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.reproducibility import seed_everything
from minifrontier.run_metadata import RunMetadata
from minifrontier.shards import PackedShardDataset, ShardBatchProvider
from minifrontier.training import (
    TrainingConfig,
    TrainingState,
    WarmupCosineSchedule,
    build_adamw,
    train_updates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--attention-impl", choices=("auto", "manual", "sdpa", "flex"))
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend")
    parser.add_argument("--compile-fail", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[TrainingState, RunMetadata]:
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    model_config = ModelConfig.from_toml(args.config)
    train_config = TrainingConfig(
        max_updates=args.updates,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_updates=args.warmup_updates,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        gradient_accumulation_steps=args.accumulation_steps,
        precision=args.precision,
        activation_checkpointing=args.activation_checkpointing,
        attention_impl=args.attention_impl,
    )
    device = torch.device(args.device)
    seed_everything(args.seed)
    model = MiniFrontier(model_config).to(device)
    dataset = PackedShardDataset(args.train_shards)
    provider = ShardBatchProvider(dataset, batch_size=args.batch_size, seed=args.seed)
    optimizer = build_adamw(model, train_config)[0]
    schedule = WarmupCosineSchedule(train_config)
    state = TrainingState()
    if args.resume is not None:
        trainer_values, cursor = load_training_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=schedule,
            trusted_local_state=True,
        )
        if trainer_values.get("training_config") != asdict(train_config):
            raise ValueError("resume training configuration does not match the checkpoint")
        state = TrainingState.from_dict(trainer_values["training_state"])
        provider.load_state_dict(cursor)
    execution_model, compile_report = maybe_compile(
        model,
        enabled=args.compile,
        path="training",
        backend=args.compile_backend,
        fail_on_error=args.compile_fail,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    def checkpoint_callback(
        current_model: MiniFrontier,
        current_optimizer: torch.optim.Optimizer,
        current_schedule: WarmupCosineSchedule,
        current_state: TrainingState,
    ) -> None:
        if current_state.completed_updates % args.checkpoint_interval:
            return
        save_training_checkpoint(
            args.output / f"checkpoint-{current_state.completed_updates:08d}",
            current_model,
            optimizer=current_optimizer,
            scheduler=current_schedule,
            trainer_state={
                "training_state": current_state.to_dict(),
                "training_config": asdict(train_config),
                "compile_report": asdict(compile_report),
            },
            data_cursor=provider.state_dict(),
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # `state.consumed_target_tokens` is cumulative and survives resume, but `elapsed`
    # only spans this process's wall time. Dividing the former by the latter after a
    # resume would silently count tokens processed by an earlier, separate process
    # against this run's clock -- reporting an inflated throughput that has nothing to
    # do with this process's actual speed.
    tokens_before_this_run = state.consumed_target_tokens
    started = time.perf_counter()
    optimizer, schedule, state, policy = train_updates(
        model,
        provider,
        train_config,
        device=device,
        optimizer=optimizer,
        schedule=schedule,
        state=state,
        update_callback=checkpoint_callback,
        forward_model=execution_model,
    )
    elapsed = time.perf_counter() - started
    final_path = args.output / "final"
    save_training_checkpoint(
        final_path,
        model,
        optimizer=optimizer,
        scheduler=schedule,
        trainer_state={
            "training_state": state.to_dict(),
            "training_config": asdict(train_config),
            "compile_report": asdict(compile_report),
            "precision": asdict(policy),
        },
        data_cursor=provider.state_dict(),
    )
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    total_vram = (
        torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    )
    peak_mb = peak_allocated / (1024**2)
    metadata = RunMetadata(
        name=f"pretrain-{model_config.preset}",
        config={
            "model": model_config.to_dict(),
            "training": asdict(train_config),
            "data_order": {"seed": args.seed, "shuffle": True, "version": 2},
        },
        seed=args.seed,
        parameters=model.parameter_count(),
        train_tokens=state.consumed_target_tokens,
        wall_seconds=elapsed,
        peak_memory_mb=peak_mb,
        tokens_per_second=(state.consumed_target_tokens - tokens_before_this_run) / elapsed,
        train_loss=state.last_loss,
        metrics={
            "completed_updates": float(state.completed_updates),
            "peak_allocated_vram_bytes": float(peak_allocated),
            "peak_reserved_vram_bytes": float(peak_reserved),
            "total_vram_bytes": float(total_vram),
        },
    )
    metadata.write_json(args.output / "run.json")
    return state, metadata


def main() -> None:
    state, metadata = run(parse_args())
    print(
        f"completed {state.completed_updates} updates, "
        f"loss={state.last_loss:.6f}, tokens/s={metadata.tokens_per_second:.1f}"
    )


if __name__ == "__main__":
    main()
