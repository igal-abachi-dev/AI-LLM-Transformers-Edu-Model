"""Export a trusted local training checkpoint as a safe model release."""

# Turns a training checkpoint (a save-game: weights plus optimizer state plus a
# data cursor) into a release (weights, config, tokenizer, model card). The
# optimizer state is dropped -- nobody downloading a model needs it.
#
# "Safe" refers to the file format. Weights are written with safetensors rather
# than `torch.save`, because a `.pt` file is a Python pickle and loading a pickle
# can execute whatever code its author put there. Fine for your own files;
# unacceptable for something other people download.
#
# "Trusted local checkpoint" is the mirror of that: this script reads a pickle, so
# only ever point it at a checkpoint you produced yourself.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.checkpoint import export_release, load_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.tokenizer import MiniFrontierTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-card", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        **json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    model = MiniFrontier(config)
    # Training checkpoints are explicitly local/trusted here; published releases contain only
    # safetensors and text metadata and never carry the pickle-backed optimizer state.
    load_training_checkpoint(
        args.checkpoint,
        model,
        restore_rng=False,
        trusted_local_state=True,
    )
    tokenizer = MiniFrontierTokenizer.from_directory(args.tokenizer)
    model_card = args.model_card.read_text(encoding="utf-8") if args.model_card else None
    export_release(args.output, model, tokenizer, model_card=model_card)
    print(f"exported release to {args.output}")


if __name__ == "__main__":
    main()
