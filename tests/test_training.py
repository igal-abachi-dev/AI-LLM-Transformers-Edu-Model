from __future__ import annotations

import pytest
import torch

from minifrontier.cache import KVCache
from minifrontier.checkpoint import load_training_checkpoint, save_training_checkpoint
from minifrontier.compilation import maybe_compile
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.precision import resolve_precision
from minifrontier.training import (
    ListBatchProvider,
    ShuffledBatchProvider,
    TrainingBatch,
    TrainingConfig,
    TrainingState,
    WarmupCosineSchedule,
    build_adamw,
    train_updates,
    validate_cpu_batch,
)


def test_warmup_cosine_schedule_boundaries_and_state() -> None:
    config = TrainingConfig(
        max_updates=6,
        learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_updates=2,
    )
    schedule = WarmupCosineSchedule(config)
    values = [schedule.learning_rate_for_update(index) for index in range(6)]
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
    with pytest.raises(IndexError):
        schedule.learning_rate_for_update(6)
    schedule.completed_updates = 3
    restored = WarmupCosineSchedule(config)
    restored.load_state_dict(schedule.state_dict())
    assert restored.completed_updates == 3


def test_shuffled_batch_provider_changes_epochs_and_resumes_exactly() -> None:
    batches = [TrainingBatch(torch.tensor([[index, index + 1]])) for index in range(8)]
    provider = ShuffledBatchProvider(batches, seed=71)
    first_epoch = [int(provider.next_batch().tokens[0, 0]) for _ in batches]
    assert sorted(first_epoch) == list(range(8))
    second_prefix = [int(provider.next_batch().tokens[0, 0]) for _ in range(3)]
    assert first_epoch != second_prefix + [
        int(provider.next_batch().tokens[0, 0]) for _ in range(5)
    ]

    provider = ShuffledBatchProvider(batches, seed=71)
    consumed = [int(provider.next_batch().tokens[0, 0]) for _ in range(11)]
    state = provider.state_dict()
    expected = [int(provider.next_batch().tokens[0, 0]) for _ in range(9)]
    resumed = ShuffledBatchProvider(batches, seed=71)
    resumed.load_state_dict(state)
    actual = [int(resumed.next_batch().tokens[0, 0]) for _ in range(9)]
    assert len(consumed) == 11 and actual == expected
    with pytest.raises(ValueError, match="policy"):
        ShuffledBatchProvider(batches, seed=72).load_state_dict(state)


def test_adamw_groups_exclude_norm_scales() -> None:
    model = MiniFrontier(ModelConfig.tiny_edu())
    config = TrainingConfig(max_updates=2, warmup_updates=0)
    optimizer, names = build_adamw(model, config)
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert "token_embedding.weight" in names["decay"]
    assert any(name.endswith("attention_norm.weight") for name in names["no_decay"])
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_cpu_batch_validation_rejects_bad_ids_and_all_masked() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        validate_cpu_batch(TrainingBatch(torch.tensor([[0, 8]])), vocab_size=8)
    with pytest.raises(ValueError, match="no valid"):
        validate_cpu_batch(
            TrainingBatch(torch.tensor([[1, 2, 3]]), loss_mask=torch.zeros(1, 3)),
            vocab_size=8,
        )


def test_gradient_accumulation_matches_unsplit_batch() -> None:
    torch.manual_seed(30)
    model_full = MiniFrontier(ModelConfig.tiny_edu(n_layers=1, d_model=16, n_heads=2, d_ff=32))
    model_split = MiniFrontier(model_full.config)
    model_split.load_state_dict(model_full.state_dict())
    tokens = torch.randint(0, model_full.config.vocab_size, (4, 8))
    base = dict(
        max_updates=1,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        warmup_updates=0,
        weight_decay=0.0,
        gradient_clip=1e9,
        precision="float32",
    )
    train_updates(
        model_full,
        ListBatchProvider([TrainingBatch(tokens)]),
        TrainingConfig(**base),
    )
    train_updates(
        model_split,
        ListBatchProvider([TrainingBatch(tokens[:2]), TrainingBatch(tokens[2:])]),
        TrainingConfig(**base, gradient_accumulation_steps=2),
    )
    for full, split in zip(model_full.parameters(), model_split.parameters(), strict=True):
        assert torch.allclose(full, split, atol=2e-6)


def test_activation_checkpointing_matches_loss_and_gradients() -> None:
    torch.manual_seed(31)
    config = ModelConfig.tiny_edu(n_layers=2, d_model=16, n_heads=2, d_ff=32)
    eager = MiniFrontier(config)
    checkpointed = MiniFrontier(config)
    checkpointed.load_state_dict(eager.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    eager_loss = eager(tokens, labels=tokens).loss
    checkpointed_loss = checkpointed(tokens, labels=tokens, activation_checkpointing=True).loss
    assert eager_loss is not None and checkpointed_loss is not None
    assert torch.equal(eager_loss, checkpointed_loss)
    eager_loss.backward()
    checkpointed_loss.backward()
    for left, right in zip(eager.parameters(), checkpointed.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.allclose(left.grad, right.grad, atol=1e-6)


def test_lazy_cache_uses_projected_bfloat16_dtype_on_cpu() -> None:
    torch.manual_seed(32)
    config = ModelConfig.tiny_edu(max_seq_len=8)
    model = MiniFrontier(config).eval()
    cache = KVCache.allocate(config, batch_size=1, device="cpu", capacity=4)
    assert cache.layers[0].dtype is None
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        model(torch.tensor([[1, 2]]), cache=cache)
    assert cache.layers[0].dtype == torch.bfloat16


def test_bfloat16_full_and_cached_logits_preserve_argmax() -> None:
    torch.manual_seed(33)
    # Pin one backend so this isolates cache/BF16 behavior; CUDA owns the separately
    # predeclared numerical tolerance, while CPU autocast proves finiteness and argmax.
    config = ModelConfig.tiny_modern(max_seq_len=12, attention_impl="sdpa")
    model = MiniFrontier(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    with (
        torch.no_grad(),
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
    ):
        full = model(tokens).logits
        cache = KVCache.allocate(config, batch_size=1, device="cpu", capacity=8)
        pieces = [model(tokens[:, index : index + 1], cache=cache).logits for index in range(8)]
        cached = torch.cat(pieces, dim=1)
    assert torch.isfinite(full).all() and torch.isfinite(cached).all()
    assert torch.equal(full.argmax(dim=-1), cached.argmax(dim=-1))


def test_checkpoint_resume_is_exactly_equal_to_uninterrupted_training(tmp_path) -> None:
    torch.manual_seed(34)
    model_config = ModelConfig.tiny_edu(n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=8)
    initial = MiniFrontier(model_config)
    uninterrupted = MiniFrontier(model_config)
    interrupted = MiniFrontier(model_config)
    uninterrupted.load_state_dict(initial.state_dict())
    interrupted.load_state_dict(initial.state_dict())
    batches = [
        TrainingBatch(torch.tensor([[1, 2, 3, 4, 5, 6]])),
        TrainingBatch(torch.tensor([[6, 5, 4, 3, 2, 1]])),
    ]
    train_config = TrainingConfig(
        max_updates=4,
        warmup_updates=1,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        precision="float32",
    )
    full_provider = ListBatchProvider(batches)
    train_updates(uninterrupted, full_provider, train_config)

    interrupted_provider = ListBatchProvider(batches)
    optimizer, schedule, state, _ = train_updates(
        interrupted,
        interrupted_provider,
        train_config,
        stop_after_updates=2,
    )
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(
        checkpoint,
        interrupted,
        optimizer=optimizer,
        scheduler=schedule,
        trainer_state=state.to_dict(),
        data_cursor=interrupted_provider.state_dict(),
    )

    resumed = MiniFrontier(model_config)
    resumed_optimizer, _ = build_adamw(resumed, train_config)
    resumed_schedule = WarmupCosineSchedule(train_config)
    trainer_values, cursor = load_training_checkpoint(
        checkpoint,
        resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_schedule,
        trusted_local_state=True,
    )
    resumed_state = TrainingState.from_dict(trainer_values)
    resumed_provider = ListBatchProvider(batches)
    resumed_provider.load_state_dict(cursor)
    train_updates(
        resumed,
        resumed_provider,
        train_config,
        optimizer=resumed_optimizer,
        schedule=resumed_schedule,
        state=resumed_state,
    )
    assert resumed_provider.state_dict() == full_provider.state_dict()
    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_precision_policy_and_compile_eager_backend() -> None:
    assert resolve_precision("auto", "cpu").resolved == "float32"
    assert resolve_precision("bfloat16", "cpu").resolved == "bfloat16"
    model = MiniFrontier(ModelConfig.tiny_edu(n_layers=1, d_model=16, n_heads=2, d_ff=32))
    compiled, report = maybe_compile(
        model, enabled=True, path="prefill", backend="eager", fullgraph=False
    )
    tokens = torch.randint(0, model.config.vocab_size, (1, 4))
    assert torch.allclose(model(tokens).logits, compiled(tokens).logits)
    assert report.compiled and report.path == "prefill"
