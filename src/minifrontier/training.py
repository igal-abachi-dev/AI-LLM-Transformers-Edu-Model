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

The learning rate is not constant. ``WarmupCosineSchedule`` starts it near zero
(large steps on a freshly randomized model are destructive), ramps up over the
first ``warmup_updates``, then eases back down along a cosine curve.

Everything with a ``state_dict`` in this file exists so a run can be interrupted
and resumed at exactly the token it stopped on -- see ``checkpoint.py``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from minifrontier.config import AttentionImplementation
from minifrontier.loss import next_token_loss_stats
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainingState:
        return cls(**values)


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
    schedule: WarmupCosineSchedule | None = None,
    state: TrainingState | None = None,
    validation_fn: Callable[[MiniFrontier, TrainingState], None] | None = None,
    update_callback: Callable[
        [MiniFrontier, torch.optim.Optimizer, WarmupCosineSchedule, TrainingState], None
    ]
    | None = None,
    forward_model: torch.nn.Module | None = None,
    stop_after_updates: int | None = None,
    mtp_heads: torch.nn.Module | None = None,
) -> tuple[
    torch.optim.Optimizer | CombinedOptimizer, WarmupCosineSchedule, TrainingState, PrecisionPolicy
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
    schedule = schedule or WarmupCosineSchedule(config)
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
            # Under BF16 autocast the matmuls run in half precision while the
            # sensitive reductions stay FP32. On CPU this context does nothing.
            with policy.autocast_context():
                output = execution_model(
                    tokens_device,
                    attention_impl=config.attention_impl,
                    activation_checkpointing=config.activation_checkpointing,
                    return_hidden_states=mtp_heads is not None,
                )
                loss_sum, _ = next_token_loss_stats(
                    output.logits,
                    labels_device,
                    loss_mask=mask_device,
                )
                # Everything backpropagated may include the weighted MTP
                # auxiliary term, but `loss_sum` itself (used for `last_loss`
                # below) stays the primary next-token loss alone -- keeping the
                # logged/reported loss directly comparable to a non-MTP run's.
                total_loss_sum = loss_sum
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
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        # Apply the nudge. This is the only line that changes the model's weights.
        # Under FP16, the scaler skips this step instead (leaving the parameters
        # untouched) whenever it detects an inf/nan gradient this update, then
        # shrinks the scale factor for next time -- the schedule still advances
        # below, so a skipped step still counts as one update, same as
        # nanoGPT/nanochat's convention.
        scaler.step(optimizer)
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
