"""Inspect the real attention weights a trained checkpoint produces for one prompt."""

# Runs one forward pass through the manual (non-fused) attention reference path,
# the only path that ever materializes the actual softmax probability grid --
# SDPA/FlexAttention never do, by design (that is the whole point of a fused
# kernel). For each layer/head, prints which prompt tokens one query position
# attends to most -- e.g. to check whether attention around a comment stays
# anchored to the current language's own tokens or drifts toward another one.
#
#   python scripts/inspect_attention.py --model artifacts/my-release --prompt "..."
#
# Read-only inspection: no training or generation-quality impact either way.
# A real, throwaway-diagnostic-style tool (same spirit as this project's other
# VRAM/throughput diagnostics), not a claim that this is a general interpretability
# framework -- it answers "what did this one prompt, on this one checkpoint,
# attend to," nothing more. Multi-line/code prompts have the same shell-escaping
# pitfall as `sample.py` -- pipe a real file via stdin rather than `--prompt
# "a\nb"` on the command line; see `sample.py`'s own header for why.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import minifrontier.attention as attention_module
from minifrontier.checkpoint import load_release
from minifrontier.precision import cast_model_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument(
        "--position",
        type=int,
        default=-1,
        help="which query position to inspect; negative indexes from the end "
        "(default: -1, the last prompt token)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="how many top-attended key tokens to print per head",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt:
        raise SystemExit("a non-empty --prompt or stdin value is required")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    model, tokenizer = load_release(args.model, device=args.device)
    policy = cast_model_for_inference(model, args.precision, args.device)
    if policy.fallback_reason:
        print(f"precision fallback: {policy.fallback_reason}", file=sys.stderr)

    token_ids = tokenizer.encode(prompt, add_bos=True)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    pieces = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]

    position = args.position if args.position >= 0 else len(token_ids) + args.position
    if not 0 <= position < len(token_ids):
        raise SystemExit(f"--position resolves to {position}, outside [0, {len(token_ids)})")

    # Swap in a capturing wrapper for the duration of one forward pass, then
    # restore the real function -- `CausalSelfAttention`'s manual path (same
    # module) resolves this name at call time, so reassigning it here redirects
    # every layer's call without touching the model or attention code at all.
    captured: list[torch.Tensor] = []
    real_manual_attention = attention_module.manual_scaled_dot_product_attention

    def _capturing_manual_attention(*call_args: object, **call_kwargs: object) -> torch.Tensor:
        output, weights = real_manual_attention(*call_args, **call_kwargs, return_weights=True)
        captured.append(weights.detach())
        return output

    attention_module.manual_scaled_dot_product_attention = _capturing_manual_attention
    try:
        model.eval()
        with torch.no_grad():
            model(tokens, attention_impl="manual")
    finally:
        attention_module.manual_scaled_dot_product_attention = real_manual_attention

    print(f"Prompt tokens ({len(token_ids)}): {pieces}")
    print(f"Inspecting query position {position} ({pieces[position]!r})\n")
    for layer_index, weights in enumerate(captured):
        # weights: [batch=1, heads, query_seq, key_seq]
        print(f"--- Layer {layer_index} ---")
        for head in range(weights.shape[1]):
            row = weights[0, head, position]
            top = torch.topk(row, min(args.top_k, row.shape[-1]))
            attended = [
                (pieces[index], round(value, 4))
                for index, value in zip(top.indices.tolist(), top.values.tolist(), strict=True)
            ]
            print(f"  head {head}: {attended}")


if __name__ == "__main__":
    main()
