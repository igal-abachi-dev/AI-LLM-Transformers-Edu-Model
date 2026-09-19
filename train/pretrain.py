"""Canonical single-process MiniFrontier pretraining entry point."""

# STEP 3 OF THE PIPELINE, and the one that takes the hours. This is where a
# randomly initialized model becomes one that has read a lot of text.
#
# What happens here, at a glance: build the model from a config, wire up the data
# shards from step 2, and hand both to `train_updates` in
# `src/minifrontier/training.py` -- which is the actual loop and the file to read
# if you want to understand training itself.
#
# A few things worth knowing before your first run:
#
# * A "step" means one optimizer update, not one batch. With gradient accumulation
#   several batches contribute to a single update.
# * The run is resumable. It checkpoints weights, optimizer state, schedule
#   position and the data cursor together, so an interrupted run continues exactly
#   where it stopped rather than re-reading data it has already learned from.
# * Loss is reported in nats per token. Starting value is around ln(vocab_size),
#   about 9.7 for a 16,384-token vocabulary -- that is the model guessing blind.
# * Do not start a serious run until the tests, the overfit proof, and the data
#   checks pass. Debugging at hour six is far more expensive than at minute one.

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from minifrontier.attention import set_flex_attention_compilation
from minifrontier.checkpoint import (
    load_training_checkpoint,
    prune_old_checkpoints,
    save_training_checkpoint,
)
from minifrontier.compilation import maybe_compile
from minifrontier.config import ModelConfig
from minifrontier.ema import EMAWeights
from minifrontier.evaluation.validation import batches_from_packed_shards, evaluate_token_batches
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.reproducibility import seed_everything
from minifrontier.run_metadata import RunMetadata
from minifrontier.shards import (
    CurriculumMixtureProvider,
    MixtureBatchProvider,
    PackedShardDataset,
    ShardBatchProvider,
)
from minifrontier.termination import GracefulTerminationRequested, TerminationRequestTracker
from minifrontier.tokenizer import MiniFrontierTokenizer
from minifrontier.training import (
    LearningRateSchedule,
    TrainingConfig,
    TrainingState,
    build_optimizer,
    build_schedule,
    train_updates,
    wsd_decay_start_update,
)

# Deliberately smaller than eval_checkpoint.py's own VALIDATION_BATCH_SIZE=8: that
# script always runs standalone with the whole GPU to itself, while this one runs
# validation IN-PROCESS, sequentially after a real training step, on whatever VRAM
# training's own weights/optimizer state/gradients have left over. Not independently
# measured on real tight-VRAM hardware -- a smaller default is the more conservative
# starting point, not a verified-safe one; profile with a real throwaway probe
# before trusting this on a production run already close to its VRAM ceiling.
VALIDATION_BATCH_SIZE = 4
# A full pass over a real multi-hundred-million-token validation split is not cheap
# enough to run every few hundred/thousand updates on a multi-day run -- this caps
# each periodic check to a small "canary" sample by default, not the full pool. The
# real, full-precision final validation still belongs to a separate scripts/
# eval_checkpoint.py pass (unaffected by this default, no batch limit there) -- this
# flag is about cheap, frequent-ish in-loop signal, not a replacement for that.
VALIDATION_MAX_BATCHES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--train-shards", type=Path, help="single-source training (mutually excl. with --mixture)"
    )
    parser.add_argument(
        "--mixture",
        action="append",
        metavar="NAME;SHARDS_PATH;WEIGHT",
        help="repeatable; e.g. '--mixture web;data/shards/web/train;0.8 "
        "--mixture code;data/shards/code/train;0.2' (MF-094). Mutually exclusive "
        "with --train-shards; at least two --mixture entries make a real mixture, "
        "though one is accepted as a degenerate single-source case.",
    )
    parser.add_argument(
        "--mixture-max-source-fraction",
        type=float,
        help=(
            "MF-148 (inspired by AI2 OLMo-core's SourceMixtureConfig): reject "
            "--mixture at startup if any one source's share of the total weight "
            "exceeds this fraction, e.g. 0.9 to catch an accidentally dominant "
            "source. Omit to disable (the default) -- purely a safety check, "
            "changes nothing about how batches are actually drawn."
        ),
    )
    parser.add_argument(
        "--mixture-max-repetition-ratio",
        type=float,
        help=(
            "MF-148 (inspired by AI2 OLMo-core's SourceMixtureConfig): reject "
            "--mixture at startup if any source would be drawn from, on expectation "
            "over the whole run (--updates x --accumulation-steps total batches), "
            "more than this many times its own available batch count -- catches an "
            "accidentally tiny or over-weighted source before a real multi-day run "
            "starts, rather than silently over-repeating it. Omit to disable (the "
            "default)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--attention-impl", choices=("auto", "manual", "sdpa", "flex"))
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--no-decay-embeddings",
        action="store_true",
        help=(
            "Exclude the token embedding from weight decay (TrainingConfig.decay_embeddings "
            "defaults to True, decaying it like every other >=2D parameter). Real published "
            "small-model recipes (OLMo 2 1B, SmolLM3) report this off as a stability "
            "improvement, particularly relevant here since embeddings are tied to lm_head "
            "by default -- not this project's own measured recommendation, no bounded test "
            "has compared the two on this project's own data/hardware."
        ),
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print a real progress line (update count, loss, tokens/s so far) every "
        "this many updates, in addition to the final summary. Without this, a run "
        "gives no signal at all until it finishes or is killed -- indistinguishable "
        "from a hang for anything longer than a few minutes. Set to a value >= "
        "--updates to disable.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help=(
            "Skip writing any checkpoint (interval or final); only run.json is written. "
            "For bounded benchmark/throughput runs whose weights are throwaway -- avoids "
            "writing hundreds of MB to GB of optimizer/model state that would just be "
            "deleted after reading the report."
        ),
    )
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend")
    parser.add_argument("--compile-fail", action="store_true")
    parser.add_argument(
        "--keep-last-n-checkpoints",
        type=int,
        help=(
            "After each periodic checkpoint, delete older ones beyond this many most "
            "recent -- never touches `final/`. Omit to keep every periodic checkpoint "
            "(the default), which a long run can turn into tens of GB of superseded state."
        ),
    )
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        help=(
            "Compute the primary loss over sequence chunks of this size instead of "
            "materializing full [batch, sequence, vocab_size] logits (see "
            "loss.chunked_next_token_loss_stats, MF-084). Omit to keep the original "
            "unfused path; a real memory/perf tradeoff worth enabling at large batch "
            "sizes/context lengths or a larger vocabulary, not a correctness fix, so "
            "not on by default."
        ),
    )
    parser.add_argument(
        "--z-loss-weight",
        type=float,
        default=0.0,
        help=(
            "PaLM-style logit-magnitude stability penalty, weight baked into the loss "
            "itself. Requires --loss-chunk-size to be set (see TrainingConfig)."
        ),
    )
    parser.add_argument(
        "--mtp-extra-heads",
        type=int,
        default=0,
        help=(
            "Multi-Token Prediction: number of extra heads predicting further ahead "
            "(t+2, t+3, ...), an off-by-default training-only experiment (see mtp.py, "
            "AGENTS.md). 0 disables MTP entirely. MTP head weights are saved in "
            "mtp_heads.safetensors alongside every periodic/final checkpoint and restored "
            "on --resume, same as the model and optimizer."
        ),
    )
    parser.add_argument("--mtp-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--schedule",
        choices=("cosine", "wsd"),
        default="cosine",
        help=(
            "cosine (default): warmup then one continuous cosine decay across the whole "
            "run. wsd (MF-083): warmup, flat at the peak rate, then a short cosine-shaped "
            "decay near the end -- adopted for the real release run on operational "
            "grounds (an interrupted multi-day run resumes in the flat phase with no "
            "schedule-shape change), not because it measures better than cosine."
        ),
    )
    parser.add_argument(
        "--wsd-decay-fraction",
        type=float,
        default=0.2,
        help="Only meaningful with --schedule wsd: fraction of --updates spent in the "
        "final decay phase.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        help=(
            "Track a decaying average of the model's weights alongside training (see "
            "ema.py, MF-086) -- a value in (0, 1), e.g. 0.999. Omit to disable (the "
            "default). The shadow weights are saved in ema.safetensors alongside every "
            "periodic/final checkpoint and restored on --resume, same as MTP heads."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "cautious_adamw"),
        default="adamw",
        help=(
            "adamw (default): plain decoupled AdamW. cautious_adamw (MF-083, "
            "arXiv:2411.16085): masks the update to only elements agreeing in sign "
            "with the current gradient, rescaling the effective learning rate to "
            "compensate -- a real, off-by-default, bounded-tested experiment."
        ),
    )
    parser.add_argument(
        "--cautious-xi",
        type=float,
        default=1.0,
        help="Only meaningful with --optimizer cautious_adamw: the paper's own "
        "normalization constant (their default is 1.0).",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=128,
        help=(
            "MF-148 (inspired by AI2 OLMo-core's StabilityMonitorCallback/"
            "SkipStepOptimizer): rolling window size (in updates) the always-on "
            "stability monitor keeps for its own loss/grad-norm history. Purely "
            "diagnostic by default -- see --skip-anomalous-steps to actually act "
            "on what it flags."
        ),
    )
    parser.add_argument(
        "--stability-sigma-factor",
        type=float,
        default=6.0,
        help="How many standard deviations above the rolling mean counts as "
        "anomalous (default matches OLMo-core's own real default).",
    )
    parser.add_argument(
        "--skip-anomalous-steps",
        action="store_true",
        help=(
            "MF-148: when the stability monitor flags an update (loss and/or "
            "grad norm a real statistical outlier against its own recent "
            "history), discard its gradients and skip the optimizer step "
            "entirely, same treatment as a genuinely non-finite gradient. Off "
            "by default -- without this flag, flagged updates are still "
            "counted (state.anomalous_updates) but proceed normally, so a real "
            "run can show how often this would have fired before anyone turns "
            "the action on."
        ),
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=0,
        help=(
            "Run a real held-out validation pass (cross-entropy/perplexity/bits-per-byte, "
            "see evaluation/validation.py) every this many updates; 0 disables it (the "
            "default). Requires --validation-shards. Entirely independent of "
            "--progress-interval (the plain-loss log line) -- there is no reason to set "
            "these equal, and a real validation pass costs meaningfully more per call than "
            "a log line does (see --validation-max-batches), so this should normally be a "
            "much larger number. This was previously wired into TrainingConfig/train_updates "
            "but never reachable from this CLI -- every real run before this flag existed "
            "only got training-loss signal, never periodic validation, regardless of intent."
        ),
    )
    parser.add_argument(
        "--validation-shards",
        action="append",
        metavar="NAME;SHARDS_PATH",
        help=(
            "repeatable; one held-out validation pool per mixture source, e.g. "
            "'--validation-shards web;data/shards/web/validation "
            "--validation-shards code;data/shards/code/validation'. Requires "
            "--validation-interval > 0. Each source is evaluated separately and reported "
            "individually, plus one token-weighted combined figure -- matching this "
            "project's own mixture-training shape rather than assuming a single pool."
        ),
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=VALIDATION_BATCH_SIZE,
        help=(
            f"Batch size for validation forward passes (inference-only, no gradients -- "
            f"independent of --batch-size). Controls per-batch parallelism/throughput, NOT "
            f"how much validation data gets evaluated -- see --validation-max-batches for "
            f"that. Default ({VALIDATION_BATCH_SIZE}) is smaller than "
            f"eval_checkpoint.py's own standalone default (8) since this runs in-process, "
            f"sequentially after a real training step, on whatever VRAM training's own "
            f"state has left over -- not independently measured on tight-VRAM hardware, so "
            f"treat this as a conservative starting point, not a verified-safe one."
        ),
    )
    parser.add_argument(
        "--validation-max-batches",
        type=int,
        default=VALIDATION_MAX_BATCHES,
        help=(
            f"Cap each source's validation pass to this many batches per check -- this, not "
            f"--validation-batch-size, is the real total-cost lever (total validation tokens "
            f"per check ~= --validation-batch-size x this x sequence_length). Default "
            f"({VALIDATION_MAX_BATCHES}) is a small 'canary' sample, not the full held-out "
            f"pool -- a full pass over a real multi-hundred-million-token validation split is "
            f"not cheap enough to run every --validation-interval on a multi-day run. Pass a "
            f"very large value for a real full-pool pass, or use scripts/eval_checkpoint.py "
            f"separately for that (unaffected by this default, no cap there)."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizer"),
        help="Only loaded when --validation-shards is given -- bits-per-byte needs it to "
        "decode packed tokens back to real UTF-8 bytes (see batches_from_packed_shards).",
    )
    parser.add_argument(
        "--decay-mixture",
        action="append",
        metavar="NAME;WEIGHT",
        help=(
            "repeatable; MF-095's decay-phase data curriculum -- reweights an "
            "existing --mixture source once WSD's decay phase starts, e.g. "
            "'--decay-mixture web;0.3 --decay-mixture code;0.7'. Requires "
            "--schedule wsd and --mixture (same source names, no new shard paths); "
            "sources not listed here keep their --mixture weight in the decay phase "
            "unchanged. Omit entirely to keep --mixture's weights fixed for the "
            "whole run (the default, unchanged behavior)."
        ),
    )
    return parser.parse_args()


def _parse_mixture_entry(spec: str) -> tuple[str, Path, float]:
    fields = spec.split(";")
    if len(fields) != 3:
        raise ValueError(f"--mixture must be 'name;shards_path;weight', got {spec!r}")
    name, shards_path, weight = fields
    if not name:
        raise ValueError(f"--mixture name must be non-empty: {spec!r}")
    return name, Path(shards_path), float(weight)


def _parse_validation_entry(spec: str) -> tuple[str, Path]:
    fields = spec.split(";")
    if len(fields) != 2:
        raise ValueError(f"--validation-shards must be 'name;shards_path', got {spec!r}")
    name, shards_path = fields
    if not name:
        raise ValueError(f"--validation-shards name must be non-empty: {spec!r}")
    return name, Path(shards_path)


def _parse_decay_mixture_entry(spec: str) -> tuple[str, float]:
    fields = spec.split(";")
    if len(fields) != 2:
        raise ValueError(f"--decay-mixture must be 'name;weight', got {spec!r}")
    name, weight = fields
    if not name:
        raise ValueError(f"--decay-mixture name must be non-empty: {spec!r}")
    return name, float(weight)


def _build_batch_provider(
    args: argparse.Namespace, train_config: TrainingConfig
) -> ShardBatchProvider | MixtureBatchProvider | CurriculumMixtureProvider:
    if (args.train_shards is None) == (not args.mixture):
        raise ValueError("exactly one of --train-shards or --mixture is required")
    if args.decay_mixture and not args.mixture:
        raise ValueError("--decay-mixture requires --mixture")
    if args.train_shards is not None:
        dataset = PackedShardDataset(args.train_shards)
        return ShardBatchProvider(dataset, batch_size=args.batch_size, seed=args.seed)
    entries = [_parse_mixture_entry(spec) for spec in args.mixture]
    names = [name for name, _, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"--mixture names must be unique, got {names}")
    providers = {
        name: ShardBatchProvider(
            PackedShardDataset(shards_path), batch_size=args.batch_size, seed=args.seed
        )
        for name, shards_path, _ in entries
    }
    stable_weights = {name: weight for name, _, weight in entries}
    # Total batches the whole run will actually draw -- what
    # --mixture-max-repetition-ratio's expectation is computed against.
    expected_total_batches = args.updates * args.accumulation_steps
    if not args.decay_mixture:
        return MixtureBatchProvider(
            providers,
            weights=stable_weights,
            seed=args.seed,
            max_source_fraction=args.mixture_max_source_fraction,
            max_repetition_ratio=args.mixture_max_repetition_ratio,
            expected_total_batches=expected_total_batches,
        )
    if train_config.schedule != "wsd":
        raise ValueError("--decay-mixture requires --schedule wsd")
    decay_overrides = dict(_parse_decay_mixture_entry(spec) for spec in args.decay_mixture)
    if not set(decay_overrides) <= set(stable_weights):
        raise ValueError("--decay-mixture names must be a subset of --mixture names")
    decay_weights = {**stable_weights, **decay_overrides}
    decay_phase_start_batch = wsd_decay_start_update(train_config) * args.accumulation_steps
    return CurriculumMixtureProvider(
        providers,
        stable_weights=stable_weights,
        decay_weights=decay_weights,
        decay_phase_start_batch=decay_phase_start_batch,
        seed=args.seed,
        max_source_fraction=args.mixture_max_source_fraction,
    )


def run(args: argparse.Namespace) -> tuple[TrainingState, RunMetadata]:
    if not args.no_checkpoint and args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if args.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if args.keep_last_n_checkpoints is not None and args.keep_last_n_checkpoints <= 0:
        raise ValueError("keep_last_n_checkpoints must be positive")
    if bool(args.validation_interval) != bool(args.validation_shards):
        raise ValueError("--validation-interval and --validation-shards require each other")
    model_config = ModelConfig.from_toml(args.config)
    train_config = TrainingConfig(
        max_updates=args.updates,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_updates=args.warmup_updates,
        weight_decay=args.weight_decay,
        decay_embeddings=not args.no_decay_embeddings,
        gradient_clip=args.gradient_clip,
        gradient_accumulation_steps=args.accumulation_steps,
        validation_interval=args.validation_interval,
        precision=args.precision,
        activation_checkpointing=args.activation_checkpointing,
        attention_impl=args.attention_impl,
        loss_chunk_size=args.loss_chunk_size,
        z_loss_weight=args.z_loss_weight,
        mtp_extra_heads=args.mtp_extra_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        schedule=args.schedule,
        wsd_decay_fraction=args.wsd_decay_fraction,
        ema_decay=args.ema_decay,
        optimizer=args.optimizer,
        cautious_xi=args.cautious_xi,
        stability_window=args.stability_window,
        stability_sigma_factor=args.stability_sigma_factor,
        skip_anomalous_steps=args.skip_anomalous_steps,
    )
    device = torch.device(args.device)
    seed_everything(args.seed)
    model = MiniFrontier(model_config).to(device)
    mtp_heads = None
    if args.mtp_extra_heads > 0:
        mtp_heads = MTPHeads(
            d_model=model_config.d_model,
            vocab_size=model_config.vocab_size,
            n_extra_heads=args.mtp_extra_heads,
            init_std=model_config.resolved_init_std,
        ).to(device)
    ema = EMAWeights(model, decay=args.ema_decay) if args.ema_decay is not None else None
    validation_fn = None
    if args.validation_shards:
        validation_entries = [_parse_validation_entry(spec) for spec in args.validation_shards]
        validation_names = [name for name, _ in validation_entries]
        if len(validation_names) != len(set(validation_names)):
            raise ValueError(f"--validation-shards names must be unique, got {validation_names}")
        validation_tokenizer = MiniFrontierTokenizer.from_directory(args.tokenizer)

        def validation_fn(current_model: MiniFrontier, current_state: TrainingState) -> None:
            parts: list[str] = []
            total_nll = 0.0
            total_tokens = 0
            total_bytes = 0
            for name, shards_path in validation_entries:
                dataset = PackedShardDataset(shards_path)
                batches = batches_from_packed_shards(
                    dataset,
                    validation_tokenizer,
                    batch_size=args.validation_batch_size,
                    device=device,
                )
                if args.validation_max_batches is not None:
                    batches = itertools.islice(batches, args.validation_max_batches)
                metrics = evaluate_token_batches(
                    current_model, batches, pad_id=validation_tokenizer.pad_id
                )
                parts.append(
                    f"{name}: ce={metrics.cross_entropy:.4f} ppl={metrics.perplexity:.2f} "
                    f"bpb={metrics.bits_per_byte:.4f} ({metrics.predicted_tokens} tok)"
                )
                total_nll += metrics.cross_entropy * metrics.predicted_tokens
                total_tokens += metrics.predicted_tokens
                total_bytes += metrics.utf8_bytes
            combined_ce = total_nll / total_tokens
            combined_bpb = total_nll / (math.log(2.0) * total_bytes)
            print(
                f"[validation @ update {current_state.completed_updates}] "
                + " | ".join(parts)
                + f" | combined: ce={combined_ce:.4f} ppl={math.exp(combined_ce):.2f} "
                f"bpb={combined_bpb:.4f}",
                flush=True,
            )

    provider = _build_batch_provider(args, train_config)
    optimizer = build_optimizer(model, train_config)[0]
    if mtp_heads is not None:
        optimizer.add_param_group(
            {"params": list(mtp_heads.parameters()), "weight_decay": train_config.weight_decay}
        )
    schedule = build_schedule(train_config)
    state = TrainingState()
    if args.resume is not None:
        trainer_values, cursor = load_training_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=schedule,
            trusted_local_state=True,
            mtp_heads=mtp_heads,
            ema=ema,
        )
        if trainer_values.get("training_config") != asdict(train_config):
            raise ValueError("resume training configuration does not match the checkpoint")
        state = TrainingState.from_dict(trainer_values["training_state"])
        provider.load_state_dict(cursor)
    execution_model, compile_report = maybe_compile(
        model,
        enabled=args.compile,
        path="training",
        backend=args.compile_backend,
        fail_on_error=args.compile_fail,
    )
    # Whole-model compilation above does not fuse FlexAttention's kernel (see
    # `attention.py`'s `set_flex_attention_compilation` docstring) -- this is the
    # separate, narrower switch that actually addresses it for local layers.
    set_flex_attention_compilation(args.compile)

    args.output.mkdir(parents=True, exist_ok=True)

    def save_checkpoint_and_prune(
        update_count: int,
        current_model: MiniFrontier,
        current_optimizer: torch.optim.Optimizer,
        current_schedule: LearningRateSchedule,
        current_state: TrainingState,
    ) -> Path:
        checkpoint_path = args.output / f"checkpoint-{update_count:08d}"
        save_training_checkpoint(
            checkpoint_path,
            current_model,
            optimizer=current_optimizer,
            scheduler=current_schedule,
            trainer_state={
                "training_state": current_state.to_dict(),
                "training_config": asdict(train_config),
                "compile_report": asdict(compile_report),
            },
            data_cursor=provider.state_dict(),
            mtp_heads=mtp_heads,
            ema=ema,
        )
        if args.keep_last_n_checkpoints is not None:
            prune_old_checkpoints(args.output, keep_last_n=args.keep_last_n_checkpoints)
        return checkpoint_path

    # Catches a *requested*, graceful stop (Ctrl+C in this console, or a real
    # SIGTERM on POSIX) so an emergency checkpoint can be forced before the
    # process actually exits -- see termination.py's own docstring for the
    # real, disclosed limits (it cannot, and no code anywhere can, protect
    # against a hard force-kill). Real motivation, not hypothetical: an early
    # 1B-token pre-work run for this project lost ~21,000 updates to an
    # external kill that landed between two scheduled checkpoints.
    termination = TerminationRequestTracker()
    try:
        termination.install()
    except ValueError:
        print(
            "WARNING: could not install the graceful-termination safety net (not running "
            "on the main thread) -- continuing without an emergency-checkpoint handler.",
            flush=True,
        )
        termination = None

    def update_progress_and_checkpoints(
        current_model: MiniFrontier,
        current_optimizer: torch.optim.Optimizer,
        current_schedule: LearningRateSchedule,
        current_state: TrainingState,
    ) -> None:
        # Real-time progress, deliberately NOT gated on --no-checkpoint: a
        # throwaway benchmark/throughput run is exactly the case --no-checkpoint
        # exists for, and exactly the case that most needs this -- without it, a
        # run gives no signal at all until it finishes or is killed, so a run
        # that is simply slower than expected becomes indistinguishable from a
        # genuine hang (a real, previously-hit problem, not a hypothetical one).
        if current_state.completed_updates % args.progress_interval == 0:
            elapsed = time.perf_counter() - started
            tokens_per_second = (
                (current_state.consumed_target_tokens - tokens_before_this_run) / elapsed
                if elapsed > 0
                else 0.0
            )
            line = [
                f"{current_state.completed_updates}/{args.updates} updates",
                f"loss={current_state.last_loss:.6f}",
            ]
            if current_state.last_z_loss is not None:
                line.append(f"z_loss={current_state.last_z_loss:.6f}")
            if current_state.last_mtp_loss is not None:
                line.append(f"mtp_loss={current_state.last_mtp_loss:.6f}")
            line.append(f"tokens/s={tokens_per_second:.1f}")
            line.append(f"elapsed={elapsed:.1f}s")
            print(", ".join(line), flush=True)
        # Checked every update, independent of --progress-interval/--checkpoint-interval:
        # a caught termination request takes priority over the normal periodic-interval
        # checkpoint below -- the update that just completed is the safe point to save
        # *right now*, not wait for the next interval boundary to come around.
        if termination is not None and termination.requested:
            if args.no_checkpoint:
                message = (
                    f"caught {termination.signal_name} -- stopping now (--no-checkpoint was "
                    f"set, no emergency checkpoint written) after "
                    f"{current_state.completed_updates} updates"
                )
            else:
                checkpoint_path = save_checkpoint_and_prune(
                    current_state.completed_updates,
                    current_model,
                    current_optimizer,
                    current_schedule,
                    current_state,
                )
                message = (
                    f"caught {termination.signal_name} -- saved an emergency checkpoint to "
                    f"{checkpoint_path} after {current_state.completed_updates} updates"
                )
            print(message, flush=True)
            raise GracefulTerminationRequested(message)
        if args.no_checkpoint or current_state.completed_updates % args.checkpoint_interval:
            return
        save_checkpoint_and_prune(
            current_state.completed_updates,
            current_model,
            current_optimizer,
            current_schedule,
            current_state,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # `state.consumed_target_tokens` is cumulative and survives resume, but `elapsed`
    # only spans this process's wall time. Dividing the former by the latter after a
    # resume would silently count tokens processed by an earlier, separate process
    # against this run's clock -- reporting an inflated throughput that has nothing to
    # do with this process's actual speed.
    tokens_before_this_run = state.consumed_target_tokens
    started = time.perf_counter()
    try:
        optimizer, schedule, state, policy = train_updates(
            model,
            provider,
            train_config,
            device=device,
            optimizer=optimizer,
            schedule=schedule,
            state=state,
            validation_fn=validation_fn,
            update_callback=update_progress_and_checkpoints,
            forward_model=execution_model,
            mtp_heads=mtp_heads,
            ema=ema,
        )
    finally:
        # Always restore -- whether training completed normally, a
        # GracefulTerminationRequested propagates out, or a real, unrelated
        # exception does. A stale handler left registered would otherwise leak
        # into whatever runs next in this same process (real risk for tests,
        # which call `run()` repeatedly without restarting the interpreter).
        if termination is not None:
            termination.restore()
    elapsed = time.perf_counter() - started
    if not args.no_checkpoint:
        final_path = args.output / "final"
        save_training_checkpoint(
            final_path,
            model,
            optimizer=optimizer,
            scheduler=schedule,
            trainer_state={
                "training_state": state.to_dict(),
                "training_config": asdict(train_config),
                "compile_report": asdict(compile_report),
                "precision": asdict(policy),
            },
            data_cursor=provider.state_dict(),
            mtp_heads=mtp_heads,
            ema=ema,
        )
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    total_vram = (
        torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    )
    peak_mb = peak_allocated / (1024**2)
    metadata = RunMetadata(
        name=f"pretrain-{model_config.preset}",
        config={
            "model": model_config.to_dict(),
            "training": asdict(train_config),
            "data_order": {"seed": args.seed, "shuffle": True, "version": 2},
        },
        seed=args.seed,
        parameters=model.parameter_count(),
        train_tokens=state.consumed_target_tokens,
        wall_seconds=elapsed,
        peak_memory_mb=peak_mb,
        tokens_per_second=(state.consumed_target_tokens - tokens_before_this_run) / elapsed,
        train_loss=state.last_loss,
        metrics={
            "completed_updates": float(state.completed_updates),
            "peak_allocated_vram_bytes": float(peak_allocated),
            "peak_reserved_vram_bytes": float(peak_reserved),
            "total_vram_bytes": float(total_vram),
        },
    )
    metadata.write_json(args.output / "run.json")
    return state, metadata


def main() -> None:
    try:
        state, metadata = run(parse_args())
    except GracefulTerminationRequested as termination_error:
        print(f"training stopped early: {termination_error}", flush=True)
        raise SystemExit(1) from None
    print(
        f"completed {state.completed_updates} updates, "
        f"loss={state.last_loss:.6f}, tokens/s={metadata.tokens_per_second:.1f}"
    )


if __name__ == "__main__":
    main()
