"""Bounded CPU integration comparison for frozen 50M Edu and Modern presets."""

# The same end-to-end smoke as `smoke_50m.py`, run for both presets side by side,
# so the Modern-only machinery is exercised too: GQA's narrower K/V projections,
# QK-Norm, the 3-local/1-global schedule, the FlexAttention path, and the ring
# cache on local layers.
#
# "Bounded" means it is deliberately tiny and finishes on a CPU. It reports
# engineering facts -- shapes line up, parity holds, artifacts round-trip -- and
# makes no claim about which preset is better. That question needs matched
# training runs; see `scripts/compare_releases.py`.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from minifrontier.cache import KVCache
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/m4-50m-cpu-smoke.json"))
    return parser.parse_args()


def _arm(config_path: Path, tokens: torch.Tensor) -> dict[str, object]:
    torch.manual_seed(44)
    config = ModelConfig.from_toml(config_path)
    model = MiniFrontier(config)
    output = model(tokens, labels=tokens, attention_impl="sdpa")
    assert output.loss is not None
    cache = KVCache.allocate(
        config,
        batch_size=tokens.shape[0],
        device="cpu",
        dtype=torch.float32,
        capacity=64,
    )
    return {
        "preset": config.preset,
        "parameters": model.parameter_count(),
        "initial_loss": output.loss.item(),
        "cache_allocated_bytes_at_64_tokens": cache.allocated_bytes(),
        "finite_logits": bool(torch.isfinite(output.logits).all()),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).parents[1]
    torch.manual_seed(45)
    tokens = torch.randint(0, 16_384, (1, 8))
    report = {
        "status": "engineering_smoke",
        "quality_claim": False,
        "speed_claim": False,
        "comparison": {
            "same_tokenizer": True,
            "same_tokens": True,
            "same_batch": True,
            "same_context": True,
            "token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        },
        "edu": _arm(root / "configs" / "50m-edu.toml", tokens),
        "modern": _arm(root / "configs" / "50m-modern.toml", tokens),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
