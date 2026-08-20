"""Run a matched-token AdamW-versus-first-party-Muon learning-rate sweep."""

# Is Muon actually better than AdamW here? This runs the experiment properly.
#
# The trap it avoids: the two optimizers have unrelated natural learning-rate
# scales, because Muon's update is normalized and AdamW's is not. Comparing them
# at one shared rate would measure "which optimizer happens to like this number",
# not which optimizer is better. So both get swept, and each is judged at its own
# best setting.
#
# "Matched-token" is the other half of the discipline: identical tokenizer, data,
# token budget, batch size, context length and seed, with the optimizer as the
# only variable. Change two things at once and the result explains nothing.
#
# For the maths behind Muon, read `labs/06_adamw_vs_muon.py` first.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from minifrontier.checkpoint import save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.muon import build_muon_adamw
from minifrontier.reproducibility import seed_everything
from minifrontier.shards import PackedShardDataset, ShardBatchProvider
from minifrontier.training import TrainingConfig, build_adamw, train_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--adamw-lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--muon-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--muon-adamw-lr", type=float, default=3e-4)
    parser.add_argument("--match-rms-adamw", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("updates and batch size must be positive")
    config = ModelConfig.from_toml(args.config)
    dataset = PackedShardDataset(args.train_shards)
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_everything(seed, deterministic=args.device == "cpu")
        initial = MiniFrontier(config).state_dict()
        arms = [("adamw", lr) for lr in args.adamw_lrs] + [("muon", lr) for lr in args.muon_lrs]
        for optimizer_name, learning_rate in arms:
            seed_everything(seed, deterministic=args.device == "cpu")
            model = MiniFrontier(config)
            model.load_state_dict(initial)
            training = TrainingConfig(
                max_updates=args.updates,
                learning_rate=learning_rate,
                min_learning_rate=learning_rate * 0.1,
                warmup_updates=min(max(args.updates // 10, 0), args.updates - 1),
                precision="float32" if args.device == "cpu" else "auto",
                attention_impl="sdpa" if args.device == "cpu" else None,
            )
            partition: dict[str, object] | None = None
            if optimizer_name == "adamw":
                optimizer = build_adamw(model, training)[0]
            else:
                optimizer, partition = build_muon_adamw(
                    model,
                    training,
                    muon_learning_rate=learning_rate,
                    adamw_learning_rate=args.muon_adamw_lr,
                    match_rms_adamw=args.match_rms_adamw,
                )
            provider = ShardBatchProvider(dataset, batch_size=args.batch_size, seed=seed)
            started = time.perf_counter()
            optimizer, schedule, state, _ = train_updates(
                model,
                provider,
                training,
                device=args.device,
                optimizer=optimizer,
            )
            elapsed = time.perf_counter() - started
            arm_name = f"seed-{seed}-{optimizer_name}-lr-{learning_rate:g}"
            save_training_checkpoint(
                args.output / arm_name,
                model,
                optimizer=optimizer,
                scheduler=schedule,
                trainer_state={
                    "training_state": state.to_dict(),
                    "training_config": asdict(training),
                    "optimizer": optimizer_name,
                    "partition": partition,
                },
                data_cursor=provider.state_dict(),
            )
            results.append(
                {
                    "arm": arm_name,
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "tokens": state.consumed_target_tokens,
                    "loss": state.last_loss,
                    "wall_seconds": elapsed,
                    "tokens_per_second": state.consumed_target_tokens / elapsed,
                    "partition": partition,
                }
            )
    token_budgets = {int(result["tokens"]) for result in results}
    if len(token_budgets) != 1:
        raise RuntimeError("optimizer arms consumed different token budgets")
    report = {
        "status": "bounded_engineering_comparison",
        "quality_claim": False,
        "model_config": config.to_dict(),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "results": results,
        "limitations": [
            "Each optimizer has its own LR sweep; no conclusion uses one shared LR.",
            "CPU bounded runs prove integration, not optimizer superiority at V1 scale.",
        ],
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
