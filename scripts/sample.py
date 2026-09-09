"""Generate a completion from a published MiniFrontier release directory."""

# The simplest way to see a trained model do something. Give it a prompt, get a
# continuation back:
#
#   python scripts/sample.py --model artifacts/my-release --prompt "2 + 2 ="
#
# This is BASE completion, not chat -- the model continues your text rather than
# answering it. That difference is entirely down to training: use `chat.py` with
# an SFT model to get replies instead.
#
# The interesting knobs are the decoding ones, and they change nothing about what
# the model knows -- only how adventurously it picks from scores it has already
# produced. `--temperature 0` (the default) is greedy and deterministic; raise it
# for variety. `--top-k` and `--top-p` restrict the candidate pool. See
# `sample_next_token` in `src/minifrontier/generation.py`.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minifrontier.chat import complete_text
from minifrontier.checkpoint import load_release, load_release_mtp_heads
from minifrontier.precision import cast_model_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-speculative",
        action="store_true",
        help=(
            "Disable self-speculative decoding even if the release has trained MTP "
            "heads (MF-093/MF-105). Speculative decoding is an exact acceleration of "
            "greedy decoding, on by default whenever a release has MTP heads and "
            "--temperature 0 --top-k None --top-p 1.0 (the default); this only exists "
            "for a plain-decode baseline, e.g. when profiling the speedup itself."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt:
        raise SystemExit("a non-empty --prompt or stdin value is required")
    model, tokenizer = load_release(args.model, device=args.device)
    policy = cast_model_for_inference(model, args.precision, args.device)
    if policy.fallback_reason:
        print(f"precision fallback: {policy.fallback_reason}", file=sys.stderr)
    # None for any Edu release, or a Modern release never trained with MTP --
    # this is what makes speculative decoding "default for Modern, never Edu"
    # a structural fact rather than a preset check: Edu has never had trained
    # MTP heads to export in the first place (AGENTS.md's MTP carve-out).
    mtp_heads = (
        None
        if args.no_speculative
        else load_release_mtp_heads(args.model, model.config, device=args.device)
    )
    print(
        complete_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            mtp_heads=mtp_heads,
        )
    )


if __name__ == "__main__":
    main()
