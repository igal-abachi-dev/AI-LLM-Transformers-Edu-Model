"""Wall-clock-matched comparison across an arbitrary number of tokenizer arms."""

# MF-087's real acceptance test, generalized for MF-090. A larger vocabulary is
# not automatically a net win just because fertility (bytes/token) improves --
# the lm_head projection itself gets more expensive too (verified: ~277 ->
# ~302 MFLOP/token for this project's 150m-modern config, roughly +9% forward
# compute, so roughly -8% tokens/second at fixed wall-clock). The only fair
# comparison is real held-out bits-per-byte (tokenizer-independent by
# construction) at *matched wall-clock time*, not matched tokens -- the same
# standard that correctly settled the Muon-vs-AdamW question
# (reports/mf070-muon-followup.md), for the same reason: whichever tokenizer
# produces more real learning per second of GPU time wins, regardless of how
# many tokens that took.
#
# N arms, each with its own tokenizer, packed shards, and ModelConfig (vocab
# size can differ across arms, so the model itself differs -- this is not a
# matched-parameter comparison, it is a matched-wall-clock-time one, same as
# the optimizer/Muon precedent). Every arm runs from a fresh model (no shared
# initial state_dict is possible here, unlike compare_mtp.py/
# compare_optimizers.py, since different-vocab arms have different embedding
# table shapes) unless its checkpoint directory already exists, in which case
# it is reused rather than retrained (see `_run_arm`'s reuse path) -- MF-090
# deliberately reuses the two checkpoints MF-087 already trained (16k, 32k
# no-leading-space) rather than re-running them for a third time.

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from minifrontier.checkpoint import load_training_checkpoint, save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.evaluation.validation import ValidationBatch, evaluate_token_batches
from minifrontier.model import MiniFrontier
from minifrontier.reproducibility import seed_everything
from minifrontier.shards import PackedShardDataset, ShardBatchProvider
from minifrontier.tokenizer import MiniFrontierTokenizer
from minifrontier.training import TrainingConfig, build_adamw, train_updates

# Smaller than compare_mtp.py's 8: at this project's larger (32k) vocabulary,
# validation's own [batch, seq, vocab_size] FP32 logits tensor is real memory
# (roughly 1GB at batch=8), stacked right on top of training's leftover
# gradients/optimizer state. Aggregate metrics are a weighted sum across
# batches regardless of batch size, so this changes nothing about the result.
VALIDATION_BATCH_SIZE = 2
# How many updates to run between wall-clock checks -- small enough that the
# budget isn't overshot by much, large enough that per-call overhead is noise.
UPDATES_PER_TICK = 25


@dataclass(frozen=True, slots=True)
class ArmSpec:
    label: str
    config_path: Path
    tokenizer_path: Path
    train_shards: Path
    validation_shards: Path


def _parse_arm_spec(spec: str) -> ArmSpec:
    """Parse ``label;config;tokenizer;train_shards;validation_shards``.

    Semicolon-delimited rather than the more common ``=``/``:`` so a Windows
    drive-letter path (``C:\\...``) in any field can never be misparsed.
    """

    fields = spec.split(";")
    if len(fields) != 5:
        raise ValueError(
            "--arm must be 'label;config;tokenizer;train_shards;validation_shards', "
            f"got {len(fields)} field(s): {spec!r}"
        )
    label, config_path, tokenizer_path, train_shards, validation_shards = fields
    if not label:
        raise ValueError(f"--arm label must be non-empty: {spec!r}")
    return ArmSpec(
        label=label,
        config_path=Path(config_path),
        tokenizer_path=Path(tokenizer_path),
        train_shards=Path(train_shards),
        validation_shards=Path(validation_shards),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        dest="arms",
        type=_parse_arm_spec,
        metavar="LABEL;CONFIG;TOKENIZER;TRAIN_SHARDS;VALIDATION_SHARDS",
        help="repeatable; one per tokenizer/config/data combination to compare",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True, help="wall-clock budget per arm")
    parser.add_argument("--max-updates", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--validation-interval-seconds",
        type=float,
        default=None,
        help=(
            "if set, run validation periodically during training (in addition to the "
            "final one) and record a curve, not just an endpoint. The time each "
            "periodic validation costs is excluded from the --seconds training budget "
            "so it stays a fair matched-training-time comparison, not a matched-"
            "wall-clock-including-validation one."
        ),
    )
    args = parser.parse_args()
    if args.validation_interval_seconds is not None and args.validation_interval_seconds <= 0:
        raise ValueError("--validation-interval-seconds must be positive when set")
    labels = [arm.label for arm in args.arms]
    if len(labels) != len(set(labels)):
        raise ValueError(f"--arm labels must be unique, got {labels}")
    return args


def _validation_batches(
    dataset: PackedShardDataset, tokenizer: MiniFrontierTokenizer, device: torch.device
) -> Iterator[ValidationBatch]:
    """Real held-out validation batches, decoded back to UTF-8 for bits-per-byte.

    Mirrors this project's own established validation recipe (compare_mtp.py,
    the MF-070 pre-work reports): pack VALIDATION_BATCH_SIZE sequences at a
    time, decode each back to text (skipping padding) purely to count real
    UTF-8 bytes -- bits-per-byte is the one metric comparable across
    tokenizers, so it is worth the decode cost.
    """

    pad_id = tokenizer.pad_id
    buffer: list[torch.Tensor] = []
    for index in range(len(dataset)):
        tokens, _ = dataset[index]
        buffer.append(tokens)
        if len(buffer) == VALIDATION_BATCH_SIZE:
            yield _stack_validation_batch(buffer, tokenizer, pad_id, device)
            buffer = []
    if buffer:
        yield _stack_validation_batch(buffer, tokenizer, pad_id, device)


def _stack_validation_batch(
    buffer: list[torch.Tensor], tokenizer: MiniFrontierTokenizer, pad_id: int, device: torch.device
) -> ValidationBatch:
    stacked = torch.stack(buffer, dim=0).to(device)
    utf8_bytes = 0
    for row in buffer:
        ids = [int(value) for value in row.tolist() if int(value) != pad_id]
        utf8_bytes += len(tokenizer.decode(ids, skip_special_tokens=True).encode("utf-8"))
    return ValidationBatch(tokens=stacked, utf8_bytes=utf8_bytes)


def _free_memory_before_validation(model: MiniFrontier, device: str) -> None:
    """Gradients from the last completed update are still populated here --

    they are only cleared at the *start* of the next update. Left alone, they
    (plus AdamW's own FP32 moment buffers, never touched here) sit fully
    resident right as validation tries to allocate its own tensors, including
    a [batch, seq, vocab_size] logits tensor at FP32. At this project's larger
    (32k) vocabulary that combination is enough to OOM an 8GB card on the very
    next small allocation -- observed for real on this run. Freeing what
    training no longer needs first is cheap and safe; it does not touch the
    optimizer's own resumable state.
    """

    model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.empty_cache()


def _run_validation(
    model: MiniFrontier,
    validation_dataset: PackedShardDataset,
    tokenizer: MiniFrontierTokenizer,
    torch_device: torch.device,
) -> dict[str, float]:
    metrics = evaluate_token_batches(
        model,
        _validation_batches(validation_dataset, tokenizer, torch_device),
        pad_id=tokenizer.pad_id,
    )
    return {
        "cross_entropy": metrics.cross_entropy,
        "perplexity": metrics.perplexity,
        "bits_per_byte": metrics.bits_per_byte,
        "predicted_tokens": metrics.predicted_tokens,
    }


@dataclass(frozen=True, slots=True)
class ArmTrainingResult:
    model: MiniFrontier
    completed_updates: int
    tokens: int
    loss: float
    training_seconds: float
    validation_series: list[dict[str, object]]


def _train_and_checkpoint_arm(
    *,
    label: str,
    config: ModelConfig,
    tokenizer: MiniFrontierTokenizer,
    train_shards: Path,
    validation_shards: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> ArmTrainingResult:
    """Real training path, with an optional periodic validation curve."""

    seed_everything(args.seed, deterministic=args.device == "cpu")
    model = MiniFrontier(config)
    training = TrainingConfig(
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        min_learning_rate=args.learning_rate * 0.1,
        warmup_updates=min(200, args.max_updates - 1),
        precision="float32" if args.device == "cpu" else "auto",
        attention_impl="sdpa" if args.device == "cpu" else None,
    )
    optimizer, names = build_adamw(model, training)
    dataset = PackedShardDataset(train_shards)
    provider = ShardBatchProvider(dataset, batch_size=args.batch_size, seed=args.seed)
    torch_device = torch.device(args.device)
    validation_dataset = (
        PackedShardDataset(validation_shards) if args.validation_interval_seconds else None
    )

    schedule = None
    state = None
    started = time.perf_counter()
    validation_overhead = 0.0
    last_periodic_validation_at = 0.0
    validation_series: list[dict[str, object]] = []
    next_stop = UPDATES_PER_TICK
    while True:
        optimizer, schedule, state, _ = train_updates(
            model,
            provider,
            training,
            device=args.device,
            optimizer=optimizer,
            schedule=schedule,
            state=state,
            stop_after_updates=next_stop,
        )
        # Training-only elapsed time: validation overhead is subtracted out so
        # --seconds always represents real training compute time, uncontaminated
        # by however many periodic validation passes ran along the way -- the
        # same "matched training time, not matched wall-clock-including-
        # validation" contract the original single-endpoint design already had.
        training_elapsed = time.perf_counter() - started - validation_overhead
        if (
            validation_dataset is not None
            and training_elapsed - last_periodic_validation_at >= args.validation_interval_seconds
        ):
            validation_started = time.perf_counter()
            _free_memory_before_validation(model, args.device)
            metrics = _run_validation(model, validation_dataset, tokenizer, torch_device)
            validation_overhead += time.perf_counter() - validation_started
            last_periodic_validation_at = training_elapsed
            validation_series.append(
                {
                    "training_elapsed_seconds": training_elapsed,
                    "completed_updates": state.completed_updates,
                    "tokens": state.consumed_target_tokens,
                    **metrics,
                }
            )
            print(
                f"{label}: t={training_elapsed:.0f}s updates={state.completed_updates} "
                f"bits_per_byte={metrics['bits_per_byte']:.4f}"
            )
        if training_elapsed >= args.seconds or state.completed_updates >= args.max_updates:
            break
        next_stop = min(state.completed_updates + UPDATES_PER_TICK, args.max_updates)
    training_elapsed = time.perf_counter() - started - validation_overhead

    _free_memory_before_validation(model, args.device)

    # Saved BEFORE the final validation, not after: validation is a real,
    # separate failure point (it OOM'd here once already), and there is no
    # reason a crash there should also cost the actual trained weights -- the
    # whole point of this comparison is the training result, not the
    # validation call. Same atomic-write path as every other checkpoint in
    # this project (see MF-079), so an interrupted save still can't corrupt
    # this one.
    save_training_checkpoint(
        checkpoint_dir,
        model,
        optimizer=optimizer,
        scheduler=schedule,
        trainer_state={
            "training_state": state.to_dict(),
            "training_config": asdict(training),
            "adamw_param_group_names": names,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        },
        data_cursor=provider.state_dict(),
    )
    return ArmTrainingResult(
        model=model,
        completed_updates=state.completed_updates,
        tokens=state.consumed_target_tokens,
        loss=state.last_loss,
        training_seconds=training_elapsed,
        validation_series=validation_series,
    )


def _run_arm(
    *,
    label: str,
    config_path: Path,
    tokenizer_path: Path,
    train_shards: Path,
    validation_shards: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    config = ModelConfig.from_toml(config_path)
    tokenizer = MiniFrontierTokenizer.from_directory(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"{label}: tokenizer vocab_size {tokenizer.vocab_size} != config vocab_size "
            f"{config.vocab_size}"
        )

    arm_name = f"seed-{args.seed}-{label}"
    checkpoint_dir = args.output / arm_name
    torch_device = torch.device(args.device)
    wall_seconds_is_exact = True
    validation_series: list[dict[str, object]] = []

    if (checkpoint_dir / "config.json").exists():
        # Reuse an existing checkpoint from an earlier, partially-completed run
        # instead of retraining -- explicit user choice, since it means
        # wall_seconds below is inferred (the target --seconds budget this arm
        # was run with), not a value this invocation actually measured itself.
        # There is no periodic validation history for a reused checkpoint --
        # only its final state is known, so validation_series stays empty.
        print(f"{arm_name}: reusing existing checkpoint at {checkpoint_dir}, skipping training")
        model = MiniFrontier(config)
        saved, _ = load_training_checkpoint(checkpoint_dir, model, trusted_local_state=True)
        model.to(torch_device)
        training_state = saved["training_state"]
        completed_updates = training_state["completed_updates"]
        consumed_tokens = training_state["consumed_target_tokens"]
        last_loss = training_state["last_loss"]
        elapsed = args.seconds
        wall_seconds_is_exact = False
    else:
        result = _train_and_checkpoint_arm(
            label=label,
            config=config,
            tokenizer=tokenizer,
            train_shards=train_shards,
            validation_shards=validation_shards,
            checkpoint_dir=checkpoint_dir,
            args=args,
        )
        model = result.model
        completed_updates = result.completed_updates
        consumed_tokens = result.tokens
        last_loss = result.loss
        elapsed = result.training_seconds
        validation_series = result.validation_series

    validation_dataset = PackedShardDataset(validation_shards)
    metrics = _run_validation(model, validation_dataset, tokenizer, torch_device)

    return {
        "arm": arm_name,
        "label": label,
        "vocab_size": config.vocab_size,
        "seed": args.seed,
        "completed_updates": completed_updates,
        "tokens": consumed_tokens,
        "tokens_per_second": consumed_tokens / elapsed,
        "loss": last_loss,
        "wall_seconds": elapsed,
        "wall_seconds_is_exact": wall_seconds_is_exact,
        "checkpoint": str(checkpoint_dir),
        "validation": metrics,
        "validation_series": validation_series,
    }


def main() -> None:
    args = parse_args()
    if args.seconds <= 0 or args.batch_size <= 0 or args.max_updates <= 0:
        raise ValueError("seconds, batch-size, and max-updates must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    results = [
        _run_arm(
            label=arm.label,
            config_path=arm.config_path,
            tokenizer_path=arm.tokenizer_path,
            train_shards=arm.train_shards,
            validation_shards=arm.validation_shards,
            args=args,
        )
        for arm in args.arms
    ]
    limitations = [
        "Single seed, single wall-clock budget -- not a sweep.",
        "Arms with different vocab sizes train different models, not matched-parameter.",
        "Bounded token budget, far short of the frozen 3B-token release target.",
    ]
    reused_arms = [result["arm"] for result in results if not result["wall_seconds_is_exact"]]
    if reused_arms:
        limitations.append(
            f"wall_seconds for {reused_arms} is the target --seconds budget, not a value "
            "this invocation measured itself -- that arm's checkpoint was reused from an "
            "earlier run rather than retrained."
        )

    report = {
        "status": "bounded_engineering_comparison",
        "quality_claim": True,
        "wall_clock_budget_seconds": args.seconds,
        "results": results,
        "limitations": limitations,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
