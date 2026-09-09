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
#
# Once the release is exported AND load-verified, the source training checkpoint
# is deleted by default: the release directory now holds everything needed for
# inference, and a training checkpoint's own weight/config copy is fully redundant
# with it. This *does* give up the ability to resume training from that exact
# checkpoint -- pass --keep-source if you still want that, e.g. exporting midway
# through a run just to sample-check progress.

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from minifrontier.checkpoint import export_release, load_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.release import verify_release
from minifrontier.tokenizer import MiniFrontierTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help=(
            "Keep the source training checkpoint (with its optimizer/RNG state) after "
            "export instead of deleting it once the release is verified loadable. "
            "Needed to resume training from this exact checkpoint later."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        **json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    model = MiniFrontier(config)
    # A checkpoint trained with MTP heads (MF-070/MF-093/MF-105) records that in its
    # own trainer_state.json -- read it directly (not a new CLI flag) so export is
    # fully automatic from what the checkpoint actually contains, and unaffected for
    # every checkpoint trained before MTP existed (missing key defaults to 0/None).
    trainer_state = json.loads((args.checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    mtp_extra_heads = trainer_state.get("training_config", {}).get("mtp_extra_heads", 0)
    mtp_heads = None
    if mtp_extra_heads > 0:
        mtp_heads = MTPHeads(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            n_extra_heads=mtp_extra_heads,
            init_std=config.resolved_init_std,
        )
    # Training checkpoints are explicitly local/trusted here; published releases contain only
    # safetensors and text metadata and never carry the pickle-backed optimizer state.
    load_training_checkpoint(
        args.checkpoint,
        model,
        restore_rng=False,
        trusted_local_state=True,
        mtp_heads=mtp_heads,
    )
    tokenizer = MiniFrontierTokenizer.from_directory(args.tokenizer)
    model_card = args.model_card.read_text(encoding="utf-8") if args.model_card else None
    export_release(args.output, model, tokenizer, model_card=model_card, mtp_heads=mtp_heads)
    # Never delete the source based on export_release() alone succeeding -- verify the
    # release is actually a complete, loadable model first. An interrupted or corrupted
    # export must never cost the only remaining copy of the trained weights.
    verify_release(args.output)
    print(f"exported release to {args.output}")
    if args.keep_source:
        return
    shutil.rmtree(args.checkpoint)
    print(f"deleted source checkpoint {args.checkpoint} (pass --keep-source to retain it)")


if __name__ == "__main__":
    main()
