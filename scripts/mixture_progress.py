"""Show real, exact per-source progress through a live run's own mixture, mid-run.

Reads a periodic training checkpoint's `training_state.pt` (the same file `--resume`
uses for exact resumability) to get each source's real `ShardBatchProvider` cursor --
not an estimate from the configured mixture weights, the actual epoch/shard/row
position each source's own provider has reached. Combined with each source's real
`manifest.json` (shard sizes only, no shard-data hashing -- fast, no multi-GB I/O),
this reconstructs exactly how many sequences/tokens have been drawn from each source
so far, and what fraction of that source's own pool that represents.

Deliberately reimplements `ShardBatchProvider`'s own tiny, deterministic shard-shuffle
formula (`seed:epoch:shards` -> sha256 -> seeded shuffle) rather than instantiating a
real `PackedShardDataset` per source, since that class's constructor re-verifies a
SHA-256 hash over every shard's actual token/count files -- correct, but multiple GB
of real disk I/O per source, too slow for a script meant to be run casually mid-run.

Run it with::

    uv run --extra cpu python scripts/mixture_progress.py \\
        --checkpoint artifacts/mf070-150m-release/checkpoint-00292000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from minifrontier.shards import ShardManifest


def _shard_shuffle_order(n_shards: int, seed: int, epoch: int) -> list[int]:
    """Exactly mirrors `ShardBatchProvider._rng`/`_reset_orders` (shards.py)."""

    label = f"{seed}:{epoch}:shards"
    derived = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    order = list(range(n_shards))
    random.Random(derived).shuffle(order)
    return order


def sequences_served(manifest: ShardManifest, provider_state: dict) -> int:
    """Real, exact lifetime sequence count a source's provider has served so far."""

    seed = provider_state["seed"]
    epoch = provider_state["epoch"]
    shard_cursor = provider_state["shard_cursor"]
    row_cursor = provider_state["row_cursor"]
    order = _shard_shuffle_order(len(manifest.shards), seed, epoch)
    completed_this_epoch = sum(manifest.shards[i].sequences for i in order[:shard_cursor])
    completed_this_epoch += row_cursor
    return epoch * manifest.total_sequences + completed_this_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="A periodic checkpoint directory"
    )
    parser.add_argument(
        "--shard-dir-template",
        default="data/shards/mf070-150m-3b-{source}",
        help="Where each source's shard pool (with manifest.json) lives; {source} is "
        "replaced with the real mixture source name. Default matches MF-070's own "
        "real layout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_state = torch.load(
        args.checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    data_cursor = local_state["data_cursor"]
    trainer_state = json.loads((args.checkpoint / "trainer_state.json").read_text())
    state = trainer_state["training_state"]
    max_updates = trainer_state["training_config"]["max_updates"]

    print(
        f"Checkpoint: {args.checkpoint.name} "
        f"({state['completed_updates']:,}/{max_updates:,} updates, "
        f"{state['completed_updates'] / max_updates:.1%})"
    )
    print(f"Total tokens consumed overall: {state['consumed_target_tokens']:,}\n")

    weights = data_cursor["weights"]
    total_weight = sum(weights.values())
    header = f"{'source':<15}{'weight':>8}{'pool_tokens':>16}{'tokens_drawn':>16}{'%_of_pool':>11}"
    print(header)
    print("-" * len(header))
    for name in sorted(data_cursor["providers"]):
        provider_state = data_cursor["providers"][name]
        manifest = ShardManifest.read(
            Path(args.shard_dir_template.format(source=name)) / "train" / "manifest.json"
        )
        seqs_served = sequences_served(manifest, provider_state)
        avg_tokens_per_seq = manifest.total_non_padding_tokens / manifest.total_sequences
        tokens_drawn = round(seqs_served * avg_tokens_per_seq)
        pct_of_pool = tokens_drawn / manifest.total_non_padding_tokens
        weight_pct = weights[name] / total_weight
        print(
            f"{name:<15}{weight_pct:>7.1%} {manifest.total_non_padding_tokens:>15,} "
            f"{tokens_drawn:>15,} {pct_of_pool:>10.1%}"
        )


if __name__ == "__main__":
    main()
