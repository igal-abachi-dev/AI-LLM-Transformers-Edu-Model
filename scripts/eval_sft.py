"""Compare base and SFT releases on the versioned original chat prompt set."""

# Did fine-tuning actually change the behaviour? Runs the same prompts through the
# base model and the SFT model side by side.
#
# The expected difference is qualitative rather than a score: the base model
# continues your question (often with more questions), while the SFT model answers
# it and stops cleanly at <|eos|>. SFT teaches format and behaviour, not knowledge
# -- everything the model knows still came from pretraining.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from minifrontier.chat import ChatMessage, generate_assistant
from minifrontier.checkpoint import load_release
from minifrontier.evaluation.sft import score_sft_responses
from minifrontier.precision import cast_model_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("eval/sft_prompts.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture_bytes = args.fixtures.read_bytes()
    fixture = json.loads(fixture_bytes)
    prompts = fixture["prompts"]
    report: dict[str, object] = {
        "status": "completed",
        "quality_claim": False,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "fixture_version": fixture["version"],
        "fixture_license": fixture["license"],
        "seed": args.seed,
        "models": {},
        "limitations": [
            "This tiny original suite detects formatting/regression failures only.",
            "Required-substring scores are transparent heuristics, not broad assistant quality.",
        ],
    }
    models = report["models"]
    assert isinstance(models, dict)
    for label, directory in (("base", args.base), ("sft", args.sft)):
        model, tokenizer = load_release(directory, device=args.device)
        policy = cast_model_for_inference(model, args.precision, args.device)
        responses = {}
        for prompt in prompts:
            messages = [ChatMessage(**message) for message in prompt["messages"]]
            responses[prompt["id"]] = generate_assistant(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )
        models[label] = {
            "precision": policy.resolved,
            "scores": score_sft_responses(prompts, responses),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
