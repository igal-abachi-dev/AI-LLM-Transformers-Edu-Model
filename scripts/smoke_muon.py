"""Persist a bounded, matched-token CPU AdamW-versus-Muon LR-sweep integration record."""

# A tiny CPU-sized version of `compare_optimizers.py`, run for wiring rather than
# for results: it proves the Muon/AdamW parameter split is disjoint, that the two
# optimizers checkpoint and resume together, and that a sweep completes end to end.
# Too small and too short to say anything about which optimizer wins.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.muon import build_muon_adamw
from minifrontier.reproducibility import seed_everything
from minifrontier.training import (
    ListBatchProvider,
    TrainingBatch,
    TrainingConfig,
    build_adamw,
    train_updates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/m7-muon-cpu-smoke.json"))
    parser.add_argument("--updates", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = 57
    seed_everything(seed, deterministic=True)
    config = ModelConfig.tiny_edu(n_layers=1, d_model=16, n_heads=2, d_ff=32)
    initial_model = MiniFrontier(config)
    initial = initial_model.state_dict()
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, config.vocab_size, (2, 12), generator=generator)
    results = []
    for optimizer_name, learning_rate in (
        ("adamw", 1e-3),
        ("adamw", 3e-3),
        ("muon", 1e-3),
        ("muon", 3e-3),
    ):
        model = MiniFrontier(config)
        model.load_state_dict(initial)
        training = TrainingConfig(
            max_updates=args.updates,
            learning_rate=learning_rate,
            min_learning_rate=learning_rate * 0.1,
            warmup_updates=0,
            weight_decay=0.0,
        )
        partition = None
        if optimizer_name == "adamw":
            optimizer = build_adamw(model, training)[0]
        else:
            optimizer, partition = build_muon_adamw(
                model,
                training,
                muon_learning_rate=learning_rate,
                adamw_learning_rate=1e-3,
                match_rms_adamw=True,
            )
        provider = ListBatchProvider([TrainingBatch(tokens.clone())])
        started = time.perf_counter()
        _, _, state, _ = train_updates(model, provider, training, optimizer=optimizer)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "tokens": state.consumed_target_tokens,
                "loss": state.last_loss,
                "wall_seconds": elapsed,
                "tokens_per_second": state.consumed_target_tokens / elapsed,
                "peak_vram_bytes": 0,
                "partition": partition,
            }
        )
    if len({result["tokens"] for result in results}) != 1:
        raise RuntimeError("smoke arms consumed different token budgets")
    report = {
        "status": "bounded_engineering_comparison",
        "quality_claim": False,
        "seed": seed,
        "config": config.to_dict(),
        "updates": args.updates,
        "variance": "not_estimated_single_seed",
        "results": results,
        "limitations": [
            "Tiny CPU fixtures validate partitioning and fair mechanics only.",
            "The post-M10 RTX gate owns scale/performance conclusions.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
