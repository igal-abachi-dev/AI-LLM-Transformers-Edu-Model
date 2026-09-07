"""MF-093: real decode-throughput comparison, plain vs self-speculative MTP decoding."""

# Loads a real trained checkpoint that has MTP heads persisted alongside it
# (see checkpoint.py's mtp_heads= parameter) and measures, on real hardware:
# plain greedy decode tokens/second, self-speculative greedy decode
# tokens/second, and the real draft-acceptance rate -- not assumed, not
# estimated from the training-time auxiliary loss.
#
# Greedy decoding is required for both arms: speculative_decoding.py's
# exactness guarantee (see its own module docstring) only holds for greedy
# decoding, so this is also, incidentally, a real-hardware correctness check --
# the two arms' output tokens must match exactly.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from minifrontier.checkpoint import load_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.generation import generate
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.speculative_decoding import speculative_generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mtp-extra-heads", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _timed(label: str, fn) -> tuple[object, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"{label}: {elapsed:.3f}s")
    return result, elapsed


def main() -> None:
    args = parse_args()
    if args.prompt_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("prompt-length and max-new-tokens must be positive")
    device = torch.device(args.device)
    config = ModelConfig.from_toml(args.checkpoint / "config.toml")
    model = MiniFrontier(config).to(device).eval()
    mtp_heads = MTPHeads(
        d_model=config.d_model,
        vocab_size=config.vocab_size,
        n_extra_heads=args.mtp_extra_heads,
    ).to(device)
    load_training_checkpoint(args.checkpoint, model, mtp_heads=mtp_heads, trusted_local_state=True)

    torch.manual_seed(args.seed)
    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_length), device=device)

    plain_tokens, plain_seconds = _timed(
        "plain",
        lambda: generate(model, prompt.clone(), max_new_tokens=args.max_new_tokens, temperature=0),
    )
    (speculative_tokens, stats), speculative_seconds = _timed(
        "speculative",
        lambda: speculative_generate(
            model, mtp_heads, prompt.clone(), max_new_tokens=args.max_new_tokens
        ),
    )

    tokens_match = torch.equal(plain_tokens, speculative_tokens)
    if not tokens_match:
        raise RuntimeError(
            "speculative decoding produced a different sequence than plain greedy decoding -- "
            "this violates the exactness guarantee and must never happen; do not trust the "
            "throughput numbers below"
        )

    plain_tokens_per_second = args.max_new_tokens / plain_seconds
    speculative_tokens_per_second = args.max_new_tokens / speculative_seconds
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "tokens_match": tokens_match,
        "plain": {"seconds": plain_seconds, "tokens_per_second": plain_tokens_per_second},
        "speculative": {
            "seconds": speculative_seconds,
            "tokens_per_second": speculative_tokens_per_second,
            "proposed": stats.proposed,
            "accepted": stats.accepted,
            "acceptance_rate": stats.acceptance_rate,
        },
        "speedup": plain_seconds / speculative_seconds,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
