"""Run a matched-token baseline-versus-Multi-Token-Prediction (MTP) comparison."""

# Does MTP's auxiliary loss (see minifrontier/mtp.py) actually help at this
# project's scale? DeepSeek-V3's real gains are reported at 671B/14.8T-token
# scale; the original MTP paper's own smaller-scale ablations show the benefit
# shrinking, sometimes reversing, well below that. This script runs the direct
# comparison rather than trusting either result to transfer.
#
# "Matched-token" discipline, as in scripts/compare_optimizers.py: identical
# tokenizer, data, token budget, batch size, context length, seed, and
# optimizer (AdamW -- the project's resolved baseline, see
# docs/IMPLEMENTATION_DECISIONS.md) between the two arms. MTP is the only
# variable. Two arms only (baseline vs one MTP configuration), not a sweep --
# there is no established "known good" MTP weight for this project yet, unlike
# the optimizer LR search compare_optimizers.py runs.
#
# For what MTP heads actually compute, read labs/08_mtp.py first.

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import torch

from minifrontier.checkpoint import save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.evaluation.validation import ValidationBatch, evaluate_token_batches
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.reproducibility import seed_everything
from minifrontier.shards import PackedShardDataset, ShardBatchProvider
from minifrontier.tokenizer import MiniFrontierTokenizer
from minifrontier.training import TrainingConfig, build_adamw, train_updates

VALIDATION_BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mtp-extra-heads", type=int, default=1)
    parser.add_argument("--mtp-loss-weight", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--validation-shards",
        type=Path,
        help="Optional packed held-out shards. Without this, only training loss is "
        "reported and quality_claim stays False, matching compare_optimizers.py.",
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    return parser.parse_args()


def _validation_batches(
    dataset: PackedShardDataset, tokenizer: MiniFrontierTokenizer, device: torch.device
) -> Iterator[ValidationBatch]:
    """Real held-out validation batches, decoded back to UTF-8 for bits-per-byte.

    Mirrors this project's own established validation recipe (see the MF-070
    pre-work reports) rather than inventing a new one: pack ``VALIDATION_BATCH_SIZE``
    sequences at a time, decode each back to text (skipping padding) purely to
    count real UTF-8 bytes -- bits-per-byte is the one metric comparable across
    tokenizers, so it is worth the decode cost.
    """

    pad_id = tokenizer.pad_id
    buffer: list[torch.Tensor] = []
    for index in range(len(dataset)):
        tokens, _ = dataset[index]
        buffer.append(tokens)
        if len(buffer) == VALIDATION_BATCH_SIZE:
            yield _stack_validation_batch(buffer, tokenizer, pad_id, device)
            buffer = []
    if buffer:
        yield _stack_validation_batch(buffer, tokenizer, pad_id, device)


def _stack_validation_batch(
    buffer: list[torch.Tensor], tokenizer: MiniFrontierTokenizer, pad_id: int, device: torch.device
) -> ValidationBatch:
    stacked = torch.stack(buffer, dim=0).to(device)
    utf8_bytes = 0
    for row in buffer:
        ids = [int(value) for value in row.tolist() if int(value) != pad_id]
        utf8_bytes += len(tokenizer.decode(ids, skip_special_tokens=True).encode("utf-8"))
    return ValidationBatch(tokens=stacked, utf8_bytes=utf8_bytes)


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("updates and batch size must be positive")
    if args.mtp_extra_heads <= 0 or args.mtp_loss_weight <= 0:
        raise ValueError("mtp-extra-heads and mtp-loss-weight must be positive")
    config = ModelConfig.from_toml(args.config)
    dataset = PackedShardDataset(args.train_shards)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = (
        MiniFrontierTokenizer.from_directory(args.tokenizer)
        if args.validation_shards is not None
        else None
    )

    seed_everything(args.seed, deterministic=args.device == "cpu")
    initial = MiniFrontier(config).state_dict()

    arms = [("baseline", 0, 0.0), ("mtp", args.mtp_extra_heads, args.mtp_loss_weight)]
    results: list[dict[str, object]] = []
    for label, mtp_extra_heads, mtp_loss_weight in arms:
        seed_everything(args.seed, deterministic=args.device == "cpu")
        model = MiniFrontier(config)
        model.load_state_dict(initial)
        training = TrainingConfig(
            max_updates=args.updates,
            learning_rate=args.learning_rate,
            min_learning_rate=args.learning_rate * 0.1,
            warmup_updates=min(max(args.updates // 10, 0), args.updates - 1),
            precision="float32" if args.device == "cpu" else "auto",
            attention_impl="sdpa" if args.device == "cpu" else None,
            mtp_extra_heads=mtp_extra_heads,
            mtp_loss_weight=mtp_loss_weight,
        )
        mtp_heads = None
        if mtp_extra_heads > 0:
            mtp_heads = MTPHeads(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                n_extra_heads=mtp_extra_heads,
            )
        # build_adamw groups the model's own parameters by decay/no-decay; MTP
        # heads are simple untied Linear layers, so they get plain weight decay
        # like the model's own Linear weights, added as one extra group.
        optimizer, names = build_adamw(model, training)
        if mtp_heads is not None:
            optimizer.add_param_group(
                {"params": list(mtp_heads.parameters()), "weight_decay": training.weight_decay}
            )
        provider = ShardBatchProvider(dataset, batch_size=args.batch_size, seed=args.seed)
        started = time.perf_counter()
        optimizer, schedule, state, _ = train_updates(
            model,
            provider,
            training,
            device=args.device,
            optimizer=optimizer,
            mtp_heads=mtp_heads,
        )
        elapsed = time.perf_counter() - started
        validation: dict[str, float | int] | None = None
        if args.validation_shards is not None:
            torch_device = torch.device(args.device)
            validation_dataset = PackedShardDataset(args.validation_shards)
            metrics = evaluate_token_batches(
                model,
                _validation_batches(validation_dataset, tokenizer, torch_device),
                pad_id=tokenizer.pad_id,
            )
            validation = {
                "cross_entropy": metrics.cross_entropy,
                "perplexity": metrics.perplexity,
                "bits_per_byte": metrics.bits_per_byte,
                "predicted_tokens": metrics.predicted_tokens,
            }
        arm_name = f"seed-{args.seed}-{label}"
        save_training_checkpoint(
            args.output / arm_name,
            model,
            optimizer=optimizer,
            scheduler=schedule,
            trainer_state={
                "training_state": state.to_dict(),
                "training_config": asdict(training),
                "mtp_extra_heads": mtp_extra_heads,
                "mtp_loss_weight": mtp_loss_weight,
                "adamw_param_group_names": names,
            },
            data_cursor=provider.state_dict(),
        )
        results.append(
            {
                "arm": arm_name,
                "label": label,
                "mtp_extra_heads": mtp_extra_heads,
                "mtp_loss_weight": mtp_loss_weight,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "tokens": state.consumed_target_tokens,
                "loss": state.last_loss,
                "wall_seconds": elapsed,
                "tokens_per_second": state.consumed_target_tokens / elapsed,
                "checkpoint": str(args.output / arm_name),
                "validation": validation,
            }
        )
    token_budgets = {int(result["tokens"]) for result in results}
    if len(token_budgets) != 1:
        raise RuntimeError("MTP arms consumed different token budgets")
    has_validation = args.validation_shards is not None
    limitations = [
        "Single seed, single MTP configuration -- not a weight/head-count sweep.",
        "MTP heads are a simplified linear-only variant, not DeepSeek-V3's full design.",
        "Bounded token budget, far short of the frozen 3B-token release target.",
    ]
    if not has_validation:
        limitations.append("No --validation-shards given: only training loss is reported.")
    report = {
        "status": "bounded_engineering_comparison",
        "quality_claim": has_validation,
        "model_config": config.to_dict(),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "results": results,
        "limitations": limitations,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
