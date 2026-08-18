"""Canonical bounded-provider AdamW training loop and resumable control state."""

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
    max_updates: int
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_updates: int = 100
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    decay_embeddings: bool = True
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    validation_interval: int = 0
    precision: Precision = "auto"
    activation_checkpointing: bool = False
    attention_impl: AttentionImplementation | None = None

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


@dataclass(slots=True)
class TrainingBatch:
    tokens: torch.Tensor
    labels: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None


class BatchProvider(Protocol):
    def next_batch(self) -> TrainingBatch: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class CombinedOptimizer:
    """Small controller for disjoint first-party optimizers with one checkpoint surface."""

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
    """Deterministic bounded provider used by tests and engineering smokes."""

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
    """Deterministic epoch-shuffled in-memory provider with compact exact-resume state."""

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
    completed_updates: int = 0
    consumed_target_tokens: int = 0
    last_loss: float | None = None
    last_gradient_norm: float | None = None
    last_learning_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainingState:
        return cls(**values)


class WarmupCosineSchedule:
    """Update-indexed warmup/cosine schedule with explicit serializable state."""

    def __init__(self, config: TrainingConfig, completed_updates: int = 0) -> None:
        if not 0 <= completed_updates <= config.max_updates:
            raise ValueError("completed_updates is outside the configured schedule")
        self.config = config
        self.completed_updates = completed_updates

    def learning_rate_for_update(self, update_index: int) -> float:
        if not 0 <= update_index < self.config.max_updates:
            raise IndexError("update index is outside the configured schedule")
        if self.config.warmup_updates and update_index < self.config.warmup_updates:
            return self.config.learning_rate * (update_index + 1) / self.config.warmup_updates
        decay_updates = self.config.max_updates - self.config.warmup_updates
        decay_index = update_index - self.config.warmup_updates
        progress = 1.0 if decay_updates <= 1 else decay_index / (decay_updates - 1)
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
    groups, names = _parameter_groups(model, config)
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    return optimizer, names


def validate_cpu_batch(batch: TrainingBatch, *, vocab_size: int) -> tuple[torch.Tensor, int]:
    """Validate IDs before CUDA transfer and return labels plus valid target count."""

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
) -> tuple[
    torch.optim.Optimizer | CombinedOptimizer, WarmupCosineSchedule, TrainingState, PrecisionPolicy
]:
    """Run explicit optimizer updates without assuming an in-memory corpus.

    ``stop_after_updates`` is an absolute update count used for bounded runs and
    deterministic interruption tests. It does not alter the serialized schedule.
    """

    torch_device = torch.device(device)
    policy = resolve_precision(config.precision, torch_device)
    optimizer = optimizer or build_adamw(model, config)[0]
    schedule = schedule or WarmupCosineSchedule(config)
    state = state or TrainingState()
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
        cpu_batches = [provider.next_batch() for _ in range(config.gradient_accumulation_steps)]
        validated = [
            validate_cpu_batch(batch, vocab_size=model.config.vocab_size) for batch in cpu_batches
        ]
        target_count = sum(count for _, count in validated)
        optimizer.zero_grad(set_to_none=True)
        detached_loss_sum = torch.zeros((), device=torch_device)
        for batch, (labels, _) in zip(cpu_batches, validated, strict=True):
            tokens_device = batch.tokens.to(torch_device)
            labels_device = labels.to(torch_device)
            mask_device = batch.loss_mask.to(torch_device) if batch.loss_mask is not None else None
            with policy.autocast_context():
                logits = execution_model(
                    tokens_device,
                    attention_impl=config.attention_impl,
                    activation_checkpointing=config.activation_checkpointing,
                ).logits
                loss_sum, _ = next_token_loss_stats(
                    logits,
                    labels_device,
                    loss_mask=mask_device,
                )
                scaled_loss = loss_sum / target_count
            scaled_loss.backward()
            detached_loss_sum += loss_sum.detach().float()

        update_index = state.completed_updates
        learning_rate = schedule.learning_rate_for_update(update_index)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        state.completed_updates += 1
        schedule.completed_updates = state.completed_updates
        state.consumed_target_tokens += target_count
        state.last_loss = (detached_loss_sum / target_count).item()
        state.last_gradient_norm = gradient_norm.item()
        state.last_learning_rate = learning_rate
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
