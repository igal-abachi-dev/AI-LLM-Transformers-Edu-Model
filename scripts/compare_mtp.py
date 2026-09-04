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
from dataclasses import asdict
from pathlib import Path

from minifrontier.checkpoint import save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.reproducibility import seed_everything
from minifrontier.shards import PackedShardDataset, ShardBatchProvider
from minifrontier.training import TrainingConfig, build_adamw, train_updates


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("updates and batch size must be positive")
    if args.mtp_extra_heads <= 0 or args.mtp_loss_weight <= 0:
        raise ValueError("mtp-extra-heads and mtp-loss-weight must be positive")
    config = ModelConfig.from_toml(args.config)
    dataset = PackedShardDataset(args.train_shards)
    args.output.mkdir(parents=True, exist_ok=True)

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
            }
        )
    token_budgets = {int(result["tokens"]) for result in results}
    if len(token_budgets) != 1:
        raise RuntimeError("MTP arms consumed different token budgets")
    report = {
        "status": "bounded_engineering_comparison",
        "quality_claim": False,
        "model_config": config.to_dict(),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "results": results,
        "limitations": [
            "Single seed, single MTP configuration -- not a weight/head-count sweep.",
            "MTP heads are a simplified linear-only variant, not DeepSeek-V3's full design.",
            "Bounded token budget, far short of the frozen 3B-token release target.",
        ],
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
