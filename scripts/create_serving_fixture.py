"""Create native greedy/logprob evidence consumed by the later vLLM/llama.cpp parity gates."""

# Records what the native PyTorch model outputs for a fixed set of prompts --
# greedy tokens and log-probabilities -- so an external runtime can later be
# checked against it.
#
# Greedy decoding is used because it is deterministic: any disagreement is then a
# real difference in the implementation, not sampling luck. This file is the
# reference answer sheet the parity gates grade against.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from minifrontier.checkpoint import load_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    model, tokenizer = load_release(args.release)
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    tokens = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens).logits.float()
        log_probabilities = logits.log_softmax(-1)
        prompt_logprobs = [
            log_probabilities[0, index - 1, token_id].item()
            for index, token_id in enumerate(prompt_ids)
            if index > 0
        ]
        generated = model.generate(tokens, max_new_tokens=args.max_new_tokens, temperature=0.0)
    completion_ids = generated[0, len(prompt_ids) :].tolist()
    report = {
        "schema_version": 1,
        "release_manifest_sha256": hashlib.sha256(
            (args.release / "sha256-manifest.json").read_bytes()
        ).hexdigest(),
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids,
        "prompt_token_logprobs": prompt_logprobs,
        "max_new_tokens": args.max_new_tokens,
        "expected_completion_token_ids": completion_ids,
        "expected_completion_text": tokenizer.decode(completion_ids),
        "precision": "float32",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote native serving parity fixture to {args.output}")


if __name__ == "__main__":
    main()
