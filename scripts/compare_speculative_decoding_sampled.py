"""MF-118: real decode-throughput comparison, plain vs self-speculative MTP
decoding, for temperature-only SAMPLING (not MF-093's greedy comparison).

Loads a real trained checkpoint with persisted MTP heads and measures, on
real hardware: plain sampled decode tokens/second, self-speculative sampled
decode tokens/second (via `speculative_generate_sampled`'s exact-rejection-
sampling path), and the real draft-acceptance rate.

Unlike `compare_speculative_decoding.py`'s greedy comparison, an exact token
match between the two arms is neither expected nor the correctness bar here:
sampling is inherently random, so even two runs of the SAME arm with
different seeds legitimately produce different output. The real correctness
claim for this technique -- that its output is distributionally, not bit-
identically, equivalent to plain sampling -- is proven separately by a
many-sample chi-squared test at the mechanism level
(`tests/test_speculative_decoding.py`); this script measures speed only.
"""

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
from minifrontier.speculative_decoding import speculative_generate_sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mtp-extra-heads", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
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
    if args.temperature <= 0:
        raise ValueError(
            "temperature must be positive -- use compare_speculative_decoding.py for greedy"
        )
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

    plain_generator = torch.Generator(device=device).manual_seed(args.seed)
    plain_tokens, plain_seconds = _timed(
        "plain",
        lambda: generate(
            model,
            prompt.clone(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            generator=plain_generator,
        ),
    )
    speculative_generator = torch.Generator(device=device).manual_seed(args.seed)
    (speculative_tokens, stats), speculative_seconds = _timed(
        "speculative",
        lambda: speculative_generate_sampled(
            model,
            mtp_heads,
            prompt.clone(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            generator=speculative_generator,
        ),
    )

    plain_tokens_per_second = args.max_new_tokens / plain_seconds
    speculative_tokens_per_second = args.max_new_tokens / speculative_seconds
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "temperature": args.temperature,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "plain": {"seconds": plain_seconds, "tokens_per_second": plain_tokens_per_second},
        "speculative": {
            "seconds": speculative_seconds,
            "tokens_per_second": speculative_tokens_per_second,
            "proposed": stats.proposed,
            "accepted": stats.accepted,
            "acceptance_rate": stats.acceptance_rate,
        },
        "speedup": plain_seconds / speculative_seconds,
        "output_lengths_equal": plain_tokens.shape == speculative_tokens.shape,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
