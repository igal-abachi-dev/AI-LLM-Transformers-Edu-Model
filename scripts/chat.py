"""Template-aware bounded-context multi-turn MiniFrontier chat CLI."""

# Interactive chat with an SFT model. Unlike `sample.py`, this wraps your text in
# the role markers the model was fine-tuned on:
#
#   <|bos|><|system|>...<|eos|><|user|>your text<|eos|><|assistant|>
#
# ...and then asks the same single question the model always answers: what token
# comes next? There is no "chat mode" inside the model -- the roles are just
# marker tokens in a flat stream. See `src/minifrontier/chat.py`.
#
# "Bounded-context" is the honest bit. The model has a small fixed window (1K-2K
# tokens here), and the whole conversation is re-sent every turn, so once it fills
# up the oldest turns get dropped. There is no memory beyond that window.

from __future__ import annotations

import argparse
from pathlib import Path

from minifrontier.chat import ChatMessage, generate_assistant, load_system_prompt
from minifrontier.checkpoint import load_release
from minifrontier.precision import cast_model_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    system = parser.add_mutually_exclusive_group()
    system.add_argument("--system", help="inline system-prompt override")
    system.add_argument("--system-file", type=Path, help="UTF-8 system-prompt override")
    system.add_argument(
        "--no-system",
        action="store_true",
        help="disable the release/default system prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_release(args.model, device=args.device)
    policy = cast_model_for_inference(model, args.precision, args.device)
    if policy.fallback_reason:
        print(f"precision fallback: {policy.fallback_reason}")
    print("Educational single-user chat; no continuous batching or server scheduling.")
    if args.no_system:
        system_prompt = None
    elif args.system is not None:
        system_prompt = args.system
    elif args.system_file is not None:
        system_prompt = load_system_prompt(args.system_file)
    else:
        release_prompt = args.model / "system_prompt.md"
        system_prompt = load_system_prompt(release_prompt if release_prompt.exists() else None)
    messages = [ChatMessage("system", system_prompt)] if system_prompt else []
    while True:
        try:
            prompt = input("user> ")
        except EOFError:
            break
        if not prompt:
            break
        messages.append(ChatMessage("user", prompt))
        try:
            reply = generate_assistant(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
        except ValueError as error:
            messages.pop()
            print(f"error: {error}")
            continue
        print(f"assistant> {reply}")
        messages.append(ChatMessage("assistant", reply or "…"))


if __name__ == "__main__":
    main()
