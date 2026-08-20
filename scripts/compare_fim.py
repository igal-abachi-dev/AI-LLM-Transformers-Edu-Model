"""Run a matched bounded baseline-versus-FIM pretraining comparison."""

# Does adding FIM examples to the mixture actually help, and what does it cost the
# ordinary left-to-right ability? Two matched runs -- same data, tokens, seed and
# everything else -- differing only in whether the FIM transform was applied.
#
# The cost side matters as much as the benefit: spending part of the training
# budget on rearranged documents is budget not spent on plain continuation, so an
# honest comparison reports both numbers. See `scripts/apply_fim.py`.

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from minifrontier.checkpoint import save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.reproducibility import seed_everything
from minifrontier.shards import PackedShardDataset, ShardBatchProvider
from minifrontier.training import TrainingConfig, train_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-shards", type=Path, required=True)
    parser.add_argument("--fim-shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=True)
    model_config = ModelConfig.from_toml(args.config)
    baseline = MiniFrontier(model_config)
    fim = MiniFrontier(model_config)
    fim.load_state_dict(baseline.state_dict())
    training_config = TrainingConfig(
        max_updates=args.updates,
        warmup_updates=min(max(args.updates // 10, 0), args.updates - 1),
        precision="float32" if args.device == "cpu" else "auto",
        attention_impl="sdpa" if args.device == "cpu" else None,
    )
    baseline_dataset = PackedShardDataset(args.baseline_shards)
    fim_dataset = PackedShardDataset(args.fim_shards)
    if baseline_dataset.manifest.sequence_length != fim_dataset.manifest.sequence_length:
        raise ValueError("matched FIM comparison requires the same packed sequence length")
    baseline_provider = ShardBatchProvider(
        baseline_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    fim_provider = ShardBatchProvider(
        fim_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    baseline_optimizer, baseline_schedule, baseline_state, _ = train_updates(
        baseline, baseline_provider, training_config, device=args.device
    )
    seed_everything(args.seed, deterministic=True)
    fim_optimizer, fim_schedule, fim_state, _ = train_updates(
        fim, fim_provider, training_config, device=args.device
    )
    if baseline_state.consumed_target_tokens != fim_state.consumed_target_tokens:
        raise ValueError("matched FIM comparison consumed different target-token budgets")
    args.output.mkdir(parents=True, exist_ok=True)
    save_training_checkpoint(
        args.output / "baseline",
        baseline,
        optimizer=baseline_optimizer,
        scheduler=baseline_schedule,
        trainer_state=baseline_state.to_dict(),
        data_cursor=baseline_provider.state_dict(),
    )
    save_training_checkpoint(
        args.output / "fim",
        fim,
        optimizer=fim_optimizer,
        scheduler=fim_schedule,
        trainer_state=fim_state.to_dict(),
        data_cursor=fim_provider.state_dict(),
    )
    report = {
        "status": "bounded_engineering_comparison",
        "quality_claim": False,
        "seed": args.seed,
        "model_config": model_config.to_dict(),
        "training_config": asdict(training_config),
        "inputs": {
            "baseline_manifest_sha256": hashlib.sha256(
                (args.baseline_shards / "manifest.json").read_bytes()
            ).hexdigest(),
            "fim_manifest_sha256": hashlib.sha256(
                (args.fim_shards / "manifest.json").read_bytes()
            ).hexdigest(),
            "sequence_length": baseline_dataset.manifest.sequence_length,
        },
        "baseline": baseline_state.to_dict(),
        "fim": fim_state.to_dict(),
        "limitations": [
            "Only the input mixture may differ between arms.",
            "A bounded CPU run proves integration and cannot establish a coding effect size.",
        ],
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
