"""Canonical bounded-provider AdamW training loop and resumable control state.

Beginner's map of this file
---------------------------
Training is a loop of four steps, repeated for as many *updates* as you budget:

1. **Forward** -- run a batch of token sequences through the model and measure how
   surprised it was by the real next tokens (``loss.py``).
2. **Backward** -- ``loss.backward()`` works out, for every single weight, which
   direction would have made the model less surprised.
3. **Clip** -- if the suggested nudge is enormous, shrink it, so one strange batch
   cannot destroy hours of progress.
4. **Step** -- the optimizer applies the nudge. AdamW is "nudge, but with memory
   of recent nudges and a per-weight step size", plus weight decay pulling weights
   gently toward zero.

Vocabulary that trips people up: a **step** here means one optimizer update, not
one batch. With ``gradient_accumulation_steps = 8`` the loop processes eight
batches, adds up their gradients, and only then updates -- which simulates a big
batch on a GPU that could not hold one.

The learning rate is not constant. Both schedules start it near zero (large
steps on a freshly randomized model are destructive) and ramp up over the
first ``warmup_updates``. ``WarmupCosineSchedule`` (the default) then eases
back down along one continuous cosine curve; ``WarmupStableDecaySchedule``
(MF-083, ``TrainingConfig.schedule="wsd"``) instead holds flat at the peak
rate until a short cosine-shaped decay near the very end -- see that class's
own docstring for why a real multi-day run prefers this shape. ``build_schedule``
is the one place that turns a ``TrainingConfig`` into the schedule it selects.

Everything with a ``state_dict`` in this file exists so a run can be interrupted
and resumed at exactly the token it stopped on -- see ``checkpoint.py``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

import torch

from minifrontier.config import AttentionImplementation
from minifrontier.loss import chunked_next_token_loss_stats, next_token_loss_stats
from minifrontier.model import MiniFrontier
from minifrontier.precision import Precision, PrecisionPolicy, resolve_precision


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Every knob of the training recipe, separate from the model's architecture."""

    # How many optimizer updates this run performs. Also sets the cosine curve's
    # length, so changing it mid-run would change the schedule shape.
    max_updates: int
    # Peak learning rate: how big a nudge each update may apply at the top of the
    # warmup ramp.
    learning_rate: float = 3e-4
    # Floor the cosine decays toward, rather than going all the way to zero.
    min_learning_rate: float = 3e-5
    # Updates spent ramping the learning rate up from ~0. Large early steps on a
    # randomly initialized model are destructive.
    warmup_updates: int = 100
    # AdamW's two memories: beta1 smooths the gradient direction, beta2 smooths its
    # magnitude. 0.95 (rather than 0.999) is the usual LLM choice -- shorter memory
    # suits a loss surface that keeps moving.
    beta1: float = 0.9
    beta2: float = 0.95
    # Pull weights gently toward zero unless the data insists otherwise. Discourages
    # memorizing individual examples.
    weight_decay: float = 0.1
    decay_embeddings: bool = True
    # Rescale the whole gradient if its total length exceeds this. The single most
    # effective guard against a loss spike wrecking a run.
    gradient_clip: float = 1.0
    # Batches to accumulate before stepping. Simulates a bigger batch than fits.
    gradient_accumulation_steps: int = 1
    # Run the validation callback every N updates; 0 disables it.
    validation_interval: int = 0
    # "auto" picks BF16 on capable CUDA and FP32 elsewhere. See precision.py.
    precision: Precision = "auto"
    # Trade extra compute for much less memory. Needed for the larger presets.
    activation_checkpointing: bool = False
    # Force one attention kernel for the whole run; None means follow the config.
    attention_impl: AttentionImplementation | None = None
    # Multi-Token Prediction, an off-by-default experiment (see mtp.py): how many
    # extra heads predict further ahead (t+2, t+3, ...) alongside the main t+1
    # head. 0 disables MTP entirely -- the default, and the only value used by
    # any released model so far.
    mtp_extra_heads: int = 0
    # How much the summed MTP auxiliary loss counts against the primary
    # next-token loss. Only meaningful when mtp_extra_heads > 0.
    mtp_loss_weight: float = 0.0
    # None (the default): compute the primary loss the original way, materializing
    # full [B, S, vocab_size] logits. A positive value switches to
    # loss.chunked_next_token_loss_stats, which never materializes that tensor --
    # see MF-084's backlog entry for why this matters once vocab_size grows. Same
    # answer either way (tested); this only trades one implementation for another.
    loss_chunk_size: int | None = None
    # PaLM-style logit-magnitude stability penalty (lambda * log_sum_exp(logits)^2),
    # weight baked in (unlike mtp_loss_weight, applied here inside the loss itself).
    # Only available alongside loss_chunk_size, since it is computed from the same
    # per-chunk log-sum-exp chunked cross-entropy already needs.
    z_loss_weight: float = 0.0
    # Which learning-rate curve `build_schedule` returns. "cosine" (the
    # default, unchanged) is `WarmupCosineSchedule`. "wsd" is
    # `WarmupStableDecaySchedule` (MF-083): warmup, then flat at the peak
    # rate, then a short decay near the end -- adopted for the real release
    # run on OPERATIONAL grounds (an interrupted multi-day run can resume
    # and keep training in the stable phase with no schedule-shape change,
    # unlike cosine, whose curve is hard-coupled to `max_updates`), not
    # because it measures better than cosine (published results put the two
    # roughly level at a similar decay fraction).
    schedule: Literal["cosine", "wsd"] = "cosine"
    # Only meaningful when schedule="wsd": the fraction of max_updates spent
    # in the final decay phase (cosine-shaped, same curve WarmupCosineSchedule
    # uses for its own tail). 0.2 matches the published result this project
    # is relying on (Hägele et al., arXiv:2405.18392) -- roughly matches
    # cosine's own quality, not a project-specific tuned value.
    wsd_decay_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.max_updates <= 0:
            raise ValueError("max_updates must be positive")
        if not 0 <= self.warmup_updates < self.max_updates:
            raise ValueError("warmup_updates must be in [0, max_updates)")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative with a positive peak")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("weight_decay must be non-negative and gradient_clip positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.validation_interval < 0:
            raise ValueError("validation_interval cannot be negative")
        if self.mtp_extra_heads < 0:
            raise ValueError("mtp_extra_heads cannot be negative")
        if self.mtp_loss_weight < 0:
            raise ValueError("mtp_loss_weight cannot be negative")
        if self.mtp_extra_heads > 0 and self.mtp_loss_weight <= 0:
            raise ValueError("mtp_loss_weight must be positive when mtp_extra_heads > 0")
        if self.mtp_extra_heads == 0 and self.mtp_loss_weight != 0.0:
            raise ValueError("mtp_loss_weight has no effect when mtp_extra_heads is 0")
        if self.loss_chunk_size is not None and self.loss_chunk_size < 1:
            raise ValueError("loss_chunk_size must be positive when provided")
        if self.z_loss_weight < 0:
            raise ValueError("z_loss_weight cannot be negative")
        if self.z_loss_weight > 0 and self.loss_chunk_size is None:
            raise ValueError("z_loss_weight requires loss_chunk_size to be set")
        if self.schedule not in ("cosine", "wsd"):
            raise ValueError(f"unknown schedule: {self.schedule}")
        if not 0.0 < self.wsd_decay_fraction <= 1.0:
            raise ValueError("wsd_decay_fraction must be in (0, 1]")


@dataclass(slots=True)
class TrainingBatch:
    """One batch: ``[batch, sequence]`` token IDs, plus optional grading rules.

    ``labels`` defaults to ``tokens`` itself, because in plain pretraining the
    answer key *is* the input -- the shift by one happens inside ``loss.py``.
    SFT supplies a ``loss_mask`` so only assistant tokens are scored.
    """

    tokens: torch.Tensor
    labels: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None


class BatchProvider(Protocol):
    """Anything that can hand out batches and say exactly where it left off.

    The two ``state_dict`` methods are what make exact resume possible: on restart
    the data stream continues from the same position rather than starting over,
    which would quietly re-train on data the model has already seen.
    """

    def next_batch(self) -> TrainingBatch: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class CombinedOptimizer:
    """Small controller for disjoint first-party optimizers with one checkpoint surface.

    Used by the Muon experiment, where different kinds of weights are handed to
    different optimizers. It makes several optimizers look like one to the training
    loop: ``zero_grad``, ``step`` and the checkpoint calls fan out to all of them.
    "Disjoint" is the safety property -- every parameter belongs to exactly one
    optimizer, checked in ``muon.partition_muon_parameters``.
    """

    def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
        if not optimizers:
            raise ValueError("at least one optimizer is required")
        self.optimizers = tuple(optimizers)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported combined optimizer state version")
        values = list(state["optimizers"])
        if len(values) != len(self.optimizers):
            raise ValueError("combined optimizer count does not match checkpoint")
        for optimizer, optimizer_state in zip(self.optimizers, values, strict=True):
            optimizer.load_state_dict(optimizer_state)


class ListBatchProvider:
    """Deterministic bounded provider used by tests and engineering smokes.

    Cycles through a fixed list forever. Perfect for proving a model *can* learn
    (see ``overfit.py``) and useless for real training, where seeing the same
    batches repeatedly is exactly what you do not want.
    """

    def __init__(self, batches: Sequence[TrainingBatch]) -> None:
        if not batches:
            raise ValueError("at least one batch is required")
        self.batches = tuple(batches)
        self.cursor = 0

    def next_batch(self) -> TrainingBatch:
        batch = self.batches[self.cursor % len(self.batches)]
        self.cursor += 1
        return batch

    def state_dict(self) -> Mapping[str, Any]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        cursor = int(state["cursor"])
        if cursor < 0:
            raise ValueError("provider cursor cannot be negative")
        self.cursor = cursor


class ShuffledBatchProvider:
    """Deterministic epoch-shuffled in-memory provider with compact exact-resume state.

    Same batches, reshuffled each pass through the data, so the model does not
    learn the order as a pattern. "Deterministic" means the same seed reproduces
    the same order exactly -- a requirement for comparing two runs honestly.
    """

    def __init__(
        self,
        batches: Sequence[TrainingBatch],
        *,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if not batches:
            raise ValueError("at least one batch is required")
        self.batches = tuple(batches)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.cursor = 0
        self._order: tuple[int, ...] = ()
        self._reset_order()

    def _reset_order(self) -> None:
        indices = list(range(len(self.batches)))
        if self.shuffle:
            # Hash sorting is stable across Python patch releases and needs no stored permutation.
            indices.sort(
                key=lambda index: (
                    hashlib.sha256(f"{self.seed}:{self.epoch}:{index}".encode()).digest(),
                    index,
                )
            )
        self._order = tuple(indices)

    def next_batch(self) -> TrainingBatch:
        batch = self.batches[self._order[self.cursor]]
        self.cursor += 1
        if self.cursor == len(self._order):
            self.epoch += 1
            self.cursor = 0
            self._reset_order()
        return batch

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "version": 1,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "batch_count": len(self.batches),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported shuffled provider state version")
        if (
            int(state["seed"]) != self.seed
            or bool(state["shuffle"]) != self.shuffle
            or int(state["batch_count"]) != len(self.batches)
        ):
            raise ValueError("shuffled provider policy does not match checkpoint")
        epoch = int(state["epoch"])
        cursor = int(state["cursor"])
        if epoch < 0 or not 0 <= cursor < len(self.batches):
            raise ValueError("invalid shuffled provider state")
        self.epoch = epoch
        self.cursor = cursor
        self._reset_order()


@dataclass(slots=True)
class TrainingState:
    """The run's progress counters -- what a checkpoint must restore to resume."""

    completed_updates: int = 0
    consumed_target_tokens: int = 0
    last_loss: float | None = None
    last_gradient_norm: float | None = None
    last_learning_rate: float | None = None
    # Only set under FP16 (see precision.py); None for BF16/FP32 runs, which never
    # use a GradScaler. Restoring this on resume matters: losing the learned scale
    # factor mid-run can reintroduce the overflow/underflow it exists to prevent.
    grad_scaler_state: dict[str, Any] | None = None
    # Under FP16, GradScaler itself detects and skips an inf/nan gradient step.
    # BF16/FP32 have no scaler to do that, so `train_updates` does the same check
    # manually and counts it here -- a run climbing this number is unwell (a data
    # or numerical-stability problem), even though each skip alone is harmless.
    nonfinite_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainingState:
        return cls(**values)


class LearningRateSchedule(Protocol):
    """Anything that maps an update index to a learning rate, exactly-resumably.

    `train_updates` only ever calls these three members -- `WarmupCosineSchedule`
    and `WarmupStableDecaySchedule` (MF-083) both satisfy this without either
    one knowing the other exists, the same duck-typed pattern `BatchProvider`
    already uses for the data-loading side of this file.
    """

    completed_updates: int

    def learning_rate_for_update(self, update_index: int) -> float: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class WarmupCosineSchedule:
    """Update-indexed warmup/cosine schedule with explicit serializable state.

    The learning rate over a run, in two phases::

        lr
         |      ___
         |    /     ---..__
         |  /              ---..__
         |/                        ----___
         +--------------------------------- update
          warmup      cosine decay

    Warmup exists because a freshly randomized model has no idea what it is doing,
    and full-size steps at that point mostly do damage. The cosine tail exists
    because late in training you want small, careful refinements.

    "Update-indexed" means the rate is a pure function of the update number, so
    resuming from a checkpoint reproduces the identical schedule -- no hidden
    counter drifting out of sync.
    """

    def __init__(self, config: TrainingConfig, completed_updates: int = 0) -> None:
        if not 0 <= completed_updates <= config.max_updates:
            raise ValueError("completed_updates is outside the configured schedule")
        self.config = config
        self.completed_updates = completed_updates

    def learning_rate_for_update(self, update_index: int) -> float:
        if not 0 <= update_index < self.config.max_updates:
            raise IndexError("update index is outside the configured schedule")
        # Warmup: a straight line from lr/warmup up to the peak.
        if self.config.warmup_updates and update_index < self.config.warmup_updates:
            return self.config.learning_rate * (update_index + 1) / self.config.warmup_updates
        decay_updates = self.config.max_updates - self.config.warmup_updates
        decay_index = update_index - self.config.warmup_updates
        # progress runs 0 -> 1 across the decay phase...
        progress = 1.0 if decay_updates <= 1 else decay_index / (decay_updates - 1)
        # ...and cos(pi * progress) turns that into a smooth 1 -> 0 fall, which is
        # then stretched between min_learning_rate and learning_rate.
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + cosine * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    def state_dict(self) -> dict[str, int]:
        return {"completed_updates": self.completed_updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        completed = int(state["completed_updates"])
        if not 0 <= completed <= self.config.max_updates:
            raise ValueError("invalid completed scheduler update count")
        self.completed_updates = completed


class WarmupStableDecaySchedule:
    """Warmup, then flat at the peak rate, then a short cosine-shaped decay.

    The learning rate over a run, in three phases (MF-083)::

        lr
         |      _____________
         |    /               \\
         |  /                  \\___
         |/
         +------------------------------ update
          warmup    stable       decay

    Adopted for the real release run on OPERATIONAL grounds, not because it
    measures better than ``WarmupCosineSchedule`` (published results put the
    two roughly level at a well-chosen decay fraction -- Hägele et al.,
    arXiv:2405.18392). The real reason: a multi-day run interrupted partway
    through the stable phase can simply resume and keep training at the flat
    rate -- the eventual decay's shape never has to change to account for how
    long the stable phase ran. Cosine cannot do this: its curve is a function
    of ``max_updates``, so training past an interruption changes the whole
    shape, including everything already trained under the old one.

    Same update-indexed, exactly-resumable design as ``WarmupCosineSchedule``:
    a pure function of the update number, so resuming from a checkpoint
    reproduces the identical schedule.
    """

    def __init__(self, config: TrainingConfig, completed_updates: int = 0) -> None:
        if not 0 <= completed_updates <= config.max_updates:
            raise ValueError("completed_updates is outside the configured schedule")
        self.config = config
        self.completed_updates = completed_updates

    def learning_rate_for_update(self, update_index: int) -> float:
        if not 0 <= update_index < self.config.max_updates:
            raise IndexError("update index is outside the configured schedule")
        # Warmup: identical to WarmupCosineSchedule's own ramp.
        if self.config.warmup_updates and update_index < self.config.warmup_updates:
            return self.config.learning_rate * (update_index + 1) / self.config.warmup_updates
        # The decay window is the LAST `wsd_decay_fraction` of the whole run,
        # not of the post-warmup remainder -- so a larger warmup does not
        # silently shrink how many updates the decay phase actually gets.
        decay_updates = max(1, round(self.config.max_updates * self.config.wsd_decay_fraction))
        decay_start = max(self.config.warmup_updates, self.config.max_updates - decay_updates)
        if update_index < decay_start:
            # Stable phase: flat at the peak rate.
            return self.config.learning_rate
        # Decay phase: the same cosine-shaped fall WarmupCosineSchedule uses
        # for its own tail, just compressed into the last `decay_updates`
        # updates instead of spanning the whole post-warmup run.
        decay_index = update_index - decay_start
        remaining_decay_updates = self.config.max_updates - decay_start
        progress = (
            1.0 if remaining_decay_updates <= 1 else decay_index / (remaining_decay_updates - 1)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + cosine * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    def state_dict(self) -> dict[str, int]:
        return {"completed_updates": self.completed_updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        completed = int(state["completed_updates"])
        if not 0 <= completed <= self.config.max_updates:
            raise ValueError("invalid completed scheduler update count")
        self.completed_updates = completed


def build_schedule(config: TrainingConfig, completed_updates: int = 0) -> LearningRateSchedule:
    """Construct whichever schedule `config.schedule` selects.

    The one place callers (CLI entry points, tests) should go to build a
    schedule from a `TrainingConfig` rather than hardcoding
    `WarmupCosineSchedule` -- MF-083 exists precisely so `config.schedule`
    actually controls what a real training run uses.
    """

    if config.schedule == "cosine":
        return WarmupCosineSchedule(config, completed_updates)
    return WarmupStableDecaySchedule(config, completed_updates)


def _parameter_groups(
    model: MiniFrontier, config: TrainingConfig
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Split weights into "decay these" and "leave these alone".

    Weight decay suits matrices, where shrinking unused directions is a genuine
    regularizer. It does not suit the 1-D parameters -- RMSNorm scales -- where
    pulling toward zero just fights the normalization the layer exists to do. The
    ``parameter.ndim >= 2`` test below is exactly that distinction.

    ``seen`` matters because tied embeddings make ``lm_head.weight`` and
    ``token_embedding.weight`` the same tensor; adding it twice would apply decay
    to it twice.
    """

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    names = {"decay": [], "no_decay": []}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        is_embedding = name == "token_embedding.weight"
        should_decay = parameter.ndim >= 2 and (config.decay_embeddings or not is_embedding)
        target = decay if should_decay else no_decay
        target_names = names["decay"] if should_decay else names["no_decay"]
        target.append(parameter)
        target_names.append(name)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return groups, names


def build_adamw(
    model: MiniFrontier, config: TrainingConfig
) -> tuple[torch.optim.AdamW, dict[str, list[str]]]:
    """Create the baseline optimizer, and report which weights landed in which group.

    The names are returned so a run record can state exactly what was decayed,
    instead of leaving it to be inferred from the code later.
    """

    groups, names = _parameter_groups(model, config)
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    return optimizer, names


def validate_cpu_batch(batch: TrainingBatch, *, vocab_size: int) -> tuple[torch.Tensor, int]:
    """Validate IDs before CUDA transfer and return labels plus valid target count.

    Why CPU-side, and why so fussy? An out-of-range token ID is a memory fault
    inside the embedding lookup on a GPU, which surfaces as an unrecoverable
    device-side assert with no useful message and usually takes the whole process
    with it. Ten microseconds of checking here turns that into a clear sentence.

    Returning the valid-target count also lets the caller weight microbatches by
    real token count instead of by batch count.
    """

    tokens = batch.tokens
    labels = batch.labels if batch.labels is not None else tokens
    if tokens.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("batch validation must run on CPU before device transfer")
    if tokens.ndim != 2 or labels.shape != tokens.shape or tokens.shape[1] < 2:
        raise ValueError("tokens and labels must share [batch, sequence>=2] shape")
    if tokens.dtype not in (torch.int32, torch.int64) or labels.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("tokens and labels must use int32 or int64")
    if int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
        raise ValueError("token ID is outside the model vocabulary")
    if batch.loss_mask is not None:
        if batch.loss_mask.device.type != "cpu" or batch.loss_mask.shape != tokens.shape:
            raise ValueError("loss_mask must be a CPU tensor matching tokens")
        valid = batch.loss_mask[:, 1:].bool() & labels[:, 1:].ne(-100)
    else:
        valid = labels[:, 1:].ne(-100)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("batch has no valid next-token targets")
    return labels, count


def train_updates(
    model: MiniFrontier,
    provider: BatchProvider,
    config: TrainingConfig,
    *,
    device: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | CombinedOptimizer | None = None,
    schedule: LearningRateSchedule | None = None,
    state: TrainingState | None = None,
    validation_fn: Callable[[MiniFrontier, TrainingState], None] | None = None,
    update_callback: Callable[
        [MiniFrontier, torch.optim.Optimizer, LearningRateSchedule, TrainingState], None
    ]
    | None = None,
    forward_model: torch.nn.Module | None = None,
    stop_after_updates: int | None = None,
    mtp_heads: torch.nn.Module | None = None,
) -> tuple[
    torch.optim.Optimizer | CombinedOptimizer, LearningRateSchedule, TrainingState, PrecisionPolicy
]:
    """Run explicit optimizer updates without assuming an in-memory corpus.

    This is the actual training loop, and it is worth reading top to bottom once:
    everything above is setup, and the ``while`` below is the whole of pretraining.

    Each pass around the loop performs ONE optimizer update, which may consume
    several batches (gradient accumulation). Data arrives through ``provider``
    rather than a list, because a real corpus is far too large to hold in memory.

    ``stop_after_updates`` is an absolute update count used for bounded runs and
    deterministic interruption tests. It does not alter the serialized schedule.

    ``mtp_heads`` is an optional ``mtp.MTPHeads`` instance for the Multi-Token
    Prediction experiment (see ``mtp.py``). It must be provided if and only if
    ``config.mtp_extra_heads > 0``; this function never constructs one itself,
    and its parameters are the caller's responsibility to include in
    ``optimizer``. ``state.last_loss`` always reports the primary next-token
    loss alone, never mixed with the MTP auxiliary term, so it stays directly
    comparable to a non-MTP run's logged loss.
    """

    if (config.mtp_extra_heads > 0) != (mtp_heads is not None):
        raise ValueError("mtp_heads must be provided if and only if config.mtp_extra_heads > 0")
    torch_device = torch.device(device)
    policy = resolve_precision(config.precision, torch_device)
    optimizer = optimizer or build_adamw(model, config)[0]
    schedule = schedule or build_schedule(config)
    state = state or TrainingState()
    # Disabled (the BF16/FP32 case) makes every scaler call below a transparent
    # no-op: .scale() returns its input unchanged, .step() just calls
    # optimizer.step(), .unscale_()/.update() do nothing. So this is safe to call
    # unconditionally rather than branching the training loop on precision.
    scaler = torch.amp.GradScaler(device=torch_device.type, enabled=policy.needs_grad_scaler)
    if policy.needs_grad_scaler and state.grad_scaler_state:
        scaler.load_state_dict(state.grad_scaler_state)
    if state.completed_updates != schedule.completed_updates:
        raise ValueError("training and scheduler update counts disagree")
    update_limit = config.max_updates if stop_after_updates is None else stop_after_updates
    if not state.completed_updates <= update_limit <= config.max_updates:
        raise ValueError("stop_after_updates must be between current and maximum updates")
    model.to(torch_device)
    model.train()
    execution_model = forward_model or model
    execution_model.train()
    if mtp_heads is not None:
        # A caller may build MTPHeads before knowing the target device (see
        # scripts/compare_mtp.py). .to() moves each Parameter's storage in
        # place rather than replacing the object, so this is safe regardless
        # of whether an optimizer already holds references to these
        # parameters -- the standard "call .to(device) before or after
        # constructing the optimizer" PyTorch guarantee.
        mtp_heads.to(torch_device)
        mtp_heads.train()
    # Computed once: MTP heads live outside `model` (see mtp.py), so their
    # gradients -- which are real, since the MTP loss feeds the same backward
    # pass below -- must be included explicitly or clip_grad_norm_ silently
    # ignores them every update.
    trainable_parameters = list(model.parameters())
    if mtp_heads is not None:
        trainable_parameters += list(mtp_heads.parameters())

    while state.completed_updates < update_limit:
        # Collect every microbatch that will contribute to this single update.
        cpu_batches = [provider.next_batch() for _ in range(config.gradient_accumulation_steps)]
        validated = [
            validate_cpu_batch(batch, vocab_size=model.config.vocab_size) for batch in cpu_batches
        ]
        # Total real (non-skipped) targets across the whole update. Dividing by this
        # -- rather than by the number of microbatches -- makes the result identical
        # to what one big batch would have produced.
        target_count = sum(count for _, count in validated)
        # Gradients accumulate by default in PyTorch, so clear last update's first.
        optimizer.zero_grad(set_to_none=True)
        detached_loss_sum = torch.zeros((), device=torch_device)
        for batch, (labels, _) in zip(cpu_batches, validated, strict=True):
            tokens_device = batch.tokens.to(torch_device)
            labels_device = labels.to(torch_device)
            mask_device = batch.loss_mask.to(torch_device) if batch.loss_mask is not None else None
            use_chunked_loss = config.loss_chunk_size is not None
            # Under BF16 autocast the matmuls run in half precision while the
            # sensitive reductions stay FP32. On CPU this context does nothing.
            with policy.autocast_context():
                output = execution_model(
                    tokens_device,
                    attention_impl=config.attention_impl,
                    activation_checkpointing=config.activation_checkpointing,
                    return_hidden_states=mtp_heads is not None or use_chunked_loss,
                    skip_logits=use_chunked_loss,
                )
                if use_chunked_loss:
                    # Never materializes [B, S, vocab_size] logits -- see
                    # loss.chunked_next_token_loss_stats and MF-084's backlog entry.
                    loss_sum, _, z_loss_sum = chunked_next_token_loss_stats(
                        output.hidden_states,
                        model.lm_head.weight,
                        labels_device,
                        loss_mask=mask_device,
                        chunk_size=config.loss_chunk_size,
                        z_loss_weight=config.z_loss_weight,
                    )
                else:
                    loss_sum, _ = next_token_loss_stats(
                        output.logits,
                        labels_device,
                        loss_mask=mask_device,
                    )
                    z_loss_sum = None
                # Everything backpropagated may include the weighted MTP
                # auxiliary term and/or z-loss, but `loss_sum` itself (used for
                # `last_loss` below) stays the primary next-token loss alone --
                # keeping the logged/reported loss directly comparable to a
                # run without either.
                total_loss_sum = loss_sum
                if z_loss_sum is not None:
                    total_loss_sum = total_loss_sum + z_loss_sum
                if mtp_heads is not None:
                    mtp_loss_sum, _ = mtp_heads.loss_sum_and_count(
                        output.hidden_states,
                        labels_device,
                        loss_mask=mask_device,
                    )
                    total_loss_sum = total_loss_sum + config.mtp_loss_weight * mtp_loss_sum
                # Pre-divided so the gradients from all microbatches sum to exactly
                # the gradient of the full batch's mean loss.
                scaled_loss = total_loss_sum / target_count
            # Compute this microbatch's gradients and ADD them to what is already
            # stored on each parameter. No optimizer step happens yet. Under FP16,
            # `scaler.scale` multiplies the loss up before backward so small
            # gradients survive FP16's narrow exponent range instead of flushing to
            # zero; the multiplication is undone below, before the gradients are
            # actually used.
            scaler.scale(scaled_loss).backward()
            # `.detach()` keeps the running total out of the autograd graph, which
            # would otherwise pin every microbatch's activations in memory.
            detached_loss_sum += loss_sum.detach().float()

        # All microbatches are in; now the single update for this iteration.
        update_index = state.completed_updates
        learning_rate = schedule.learning_rate_for_update(update_index)
        for group in optimizer.param_groups:
            # `lr_scale` lets one schedule drive two optimizers at different rates,
            # which the Muon/AdamW experiment needs. It is 1.0 for plain AdamW.
            group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))
        # Undo scaler.scale's multiplication before the gradients are read or
        # clipped -- clip_grad_norm_ and the optimizer must see real gradients, not
        # ones inflated by the FP16 scale factor. A no-op when the scaler is
        # disabled (its gradients were never scaled in the first place).
        scaler.unscale_(optimizer)
        # If the combined gradient is longer than `gradient_clip`, scale the whole
        # thing down to that length. Direction preserved, magnitude capped -- one
        # freak batch cannot then blow the model up. The returned norm is the
        # pre-clipping length, which is a useful health signal in the logs.
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, config.gradient_clip)
        # Apply the nudge. This is the only line that changes the model's weights.
        # Under FP16, the scaler itself skips this step (leaving the parameters
        # untouched) whenever it detects an inf/nan gradient this update, then
        # shrinks the scale factor for next time. BF16/FP32 have no scaler doing
        # that check, so `torch.isfinite` does it here -- without it, one batch
        # producing an inf/nan gradient would silently corrupt every weight,
        # permanently, with no warning. Either way, the schedule still advances
        # below, so a skipped step still counts as one update, same as
        # nanoGPT/nanochat's convention.
        if policy.needs_grad_scaler or torch.isfinite(gradient_norm):
            scaler.step(optimizer)
        else:
            optimizer.zero_grad(set_to_none=True)
            state.nonfinite_updates += 1
        scaler.update()
        state.completed_updates += 1
        schedule.completed_updates = state.completed_updates
        state.consumed_target_tokens += target_count
        state.last_loss = (detached_loss_sum / target_count).item()
        state.last_gradient_norm = gradient_norm.item()
        state.last_learning_rate = learning_rate
        if policy.needs_grad_scaler:
            state.grad_scaler_state = scaler.state_dict()
        if update_callback is not None:
            update_callback(model, optimizer, schedule, state)
        if (
            validation_fn is not None
            and config.validation_interval
            and state.completed_updates % config.validation_interval == 0
        ):
            validation_fn(model, state)
            model.train()
    return optimizer, schedule, state, policy
