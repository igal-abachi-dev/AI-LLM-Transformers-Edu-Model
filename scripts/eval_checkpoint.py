"""Evaluate a raw train/pretrain.py checkpoint against held-out packed shards.

Fills a real gap: scripts/eval.py only accepts a full export_release directory
(minifrontier.checkpoint.load_release), but every bounded-ablation comparison
this project runs (MF-081/082/107/108) produces raw train/pretrain.py
checkpoints, never a full release export -- a throwaway comparison arm doesn't
warrant one. batches_from_packed_shards/evaluate_token_batches is the standing
recipe (MF-086) every other bounded comparison already reuses; this script is
the one place that recipe runs on demand against an arbitrary already-trained
raw checkpoint, without retraining or exporting anything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minifrontier.checkpoint import load_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.ema import EMAWeights
from minifrontier.evaluation.language import MiniFrontierEvalLM, harness_settings
from minifrontier.evaluation.validation import batches_from_packed_shards, evaluate_token_batches
from minifrontier.model import MiniFrontier
from minifrontier.shards import PackedShardDataset
from minifrontier.tokenizer import MiniFrontierTokenizer

VALIDATION_BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="directory containing config.json"
    )
    parser.add_argument("--validation-shards", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--batch-size", type=int, default=VALIDATION_BATCH_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--weights",
        choices=("live", "ema"),
        default="live",
        help="'ema' loads the checkpoint's ema.safetensors shadow (MF-086 part 4) and "
        "evaluates it in place of the live weights -- requires the checkpoint to have "
        "been trained with ema_decay set; a checkpoint without ema.safetensors raises "
        "a clear error rather than silently falling back to live weights.",
    )
    parser.add_argument(
        "--run-harness",
        action="store_true",
        help="Also run lm-eval tasks (MiniFrontierEvalLM adapter) against this same "
        "checkpoint -- the same standing harness scripts/eval.py uses for a full "
        "release, but without needing to export one for a throwaway comparison arm.",
    )
    parser.add_argument("--include-gsm8k", action="store_true")
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Add blimp/lambada_openai/winogrande/openbookqa/commonsense_qa/boolq "
        "(MF-086) on top of DEFAULT_TASKS.",
    )
    parser.add_argument(
        "--include-cruxeval",
        action="store_true",
        help=(
            "Add cruxeval_input/cruxeval_output (MF-122) -- real code-reasoning tasks "
            "that EXECUTE the model's own generated Python (lm-eval's own cruxeval "
            "task is marked unsafe_code: true). This flag is this project's explicit "
            "consent to that execution (threaded into simple_evaluate's own "
            "confirm_run_unsafe_code gate) -- there is no sandbox beyond lm-eval's own "
            "resource-limiting guard."
        ),
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--harness-batch-size",
        type=int,
        default=8,
        help="MF-114: how many lm-eval loglikelihood/loglikelihood_rolling requests "
        "MiniFrontierEvalLM folds into one forward call. 1 reproduces the original, "
        "fully unbatched behavior exactly.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> object:
    """Preserve scalar metrics from NumPy/Torch and stringify harness metadata objects."""

    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


def evaluate_checkpoint(
    checkpoint: Path,
    validation_shards: Path,
    tokenizer_dir: Path,
    *,
    batch_size: int = VALIDATION_BATCH_SIZE,
    device: str = "cpu",
    run_harness: bool = False,
    include_gsm8k: bool = False,
    include_extended: bool = False,
    include_cruxeval: bool = False,
    harness_limit: int = 10,
    harness_batch_size: int = 8,
    weights: str = "live",
) -> dict[str, object]:
    if weights not in ("live", "ema"):
        raise ValueError(f"unknown weights selection: {weights!r}")
    config = ModelConfig(**json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    model = MiniFrontier(config).to(device)
    ema = EMAWeights(model, decay=0.999) if weights == "ema" else None
    load_training_checkpoint(checkpoint, model, trusted_local_state=True, ema=ema)
    if ema is not None:
        # Overwrite the just-loaded live weights with the EMA shadow in place --
        # everything below (eval, harness) then sees the EMA model, not live.
        ema.copy_to(model)
    tokenizer = MiniFrontierTokenizer.from_directory(tokenizer_dir)
    dataset = PackedShardDataset(validation_shards)
    metrics = evaluate_token_batches(
        model,
        batches_from_packed_shards(dataset, tokenizer, batch_size=batch_size, device=device),
        pad_id=tokenizer.pad_id,
    )
    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "weights": weights,
        **asdict(metrics),
    }
    if run_harness:
        from lm_eval import simple_evaluate

        settings = harness_settings(
            include_gsm8k=include_gsm8k,
            include_extended=include_extended,
            include_cruxeval=include_cruxeval,
        )
        adapter = MiniFrontierEvalLM(model, tokenizer, eval_batch_size=harness_batch_size)
        try:
            result["harness"] = {
                "status": "completed",
                "settings": settings,
                "results": simple_evaluate(
                    model=adapter,
                    tasks=settings["tasks"],
                    num_fewshot=0,
                    limit=harness_limit,
                    log_samples=False,
                    confirm_run_unsafe_code=settings["confirm_run_unsafe_code"],
                ),
            }
        except Exception as error:  # preserve infrastructure failure separately from scores
            result["harness"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        args.validation_shards,
        args.tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        run_harness=args.run_harness,
        include_gsm8k=args.include_gsm8k,
        include_extended=args.include_extended,
        include_cruxeval=args.include_cruxeval,
        harness_limit=args.limit,
        harness_batch_size=args.harness_batch_size,
        weights=args.weights,
    )
    text = json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
