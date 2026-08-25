from __future__ import annotations

import math

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


# --- CUDA-only tests: MF-046 (BF16 autocast) and MF-049 (activation checkpointing) -----------
#
# Everything above this line runs identically on CPU-only CI. The tests below require a real CUDA
# device and skip cleanly without one. Tolerances are declared here, before any test has been run
# against this hardware, per MF-046/049's "tolerances fixed before measurement" requirement -- they
# are not tuned after the fact to make a run pass.

_CUDA_BF16_LOGIT_ATOL = 5e-2  # BF16 keeps ~3 decimal digits of mantissa precision.
_CUDA_FP32_GRAD_ATOL = 1e-5  # cuDNN/cuBLAS reductions aren't bit-identical to CPU's 1e-6 baseline.

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@requires_cuda
def test_cuda_bfloat16_full_and_cached_logits_match_within_declared_tolerance() -> None:
    torch.manual_seed(40)
    device = torch.device("cuda")
    config = ModelConfig.tiny_modern(max_seq_len=12, attention_impl="sdpa")
    model = MiniFrontier(config).to(device).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 8), device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full = model(tokens).logits
        cache = KVCache.allocate(config, batch_size=1, device=device, capacity=8)
        assert cache.layers[0].dtype is None  # lazy: not inferred from the embedding output
        pieces = [model(tokens[:, index : index + 1], cache=cache).logits for index in range(8)]
        cached = torch.cat(pieces, dim=1)
    # The cache adopted BF16 because that is what the BF16-autocast projections actually produced.
    assert cache.layers[0].dtype == torch.bfloat16
    assert torch.isfinite(full).all() and torch.isfinite(cached).all()
    assert torch.allclose(full, cached, atol=_CUDA_BF16_LOGIT_ATOL)
    assert torch.equal(full.argmax(dim=-1), cached.argmax(dim=-1))


@requires_cuda
def test_cuda_gradient_accumulation_matches_unsplit_batch() -> None:
    torch.manual_seed(41)
    device = torch.device("cuda")
    model_full = MiniFrontier(ModelConfig.tiny_edu(n_layers=1, d_model=16, n_heads=2, d_ff=32)).to(
        device
    )
    model_split = MiniFrontier(model_full.config).to(device)
    model_split.load_state_dict(model_full.state_dict())
    # Batch validation runs on CPU by design (see validate_cpu_batch); train_updates
    # moves tokens/labels to `device` itself, so the batch must start out on CPU.
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
        device=device,
    )
    train_updates(
        model_split,
        ListBatchProvider([TrainingBatch(tokens[:2]), TrainingBatch(tokens[2:])]),
        TrainingConfig(**base, gradient_accumulation_steps=2),
        device=device,
    )
    for full, split in zip(model_full.parameters(), model_split.parameters(), strict=True):
        assert torch.allclose(full, split, atol=_CUDA_FP32_GRAD_ATOL)


@requires_cuda
def test_cuda_activation_checkpointing_matches_loss_and_gradients() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    config = ModelConfig.tiny_edu(n_layers=2, d_model=16, n_heads=2, d_ff=32)
    eager = MiniFrontier(config).to(device)
    checkpointed = MiniFrontier(config).to(device)
    checkpointed.load_state_dict(eager.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, 8), device=device)
    eager_loss = eager(tokens, labels=tokens).loss
    checkpointed_loss = checkpointed(tokens, labels=tokens, activation_checkpointing=True).loss
    assert eager_loss is not None and checkpointed_loss is not None
    assert torch.allclose(eager_loss, checkpointed_loss, atol=_CUDA_FP32_GRAD_ATOL)
    eager_loss.backward()
    checkpointed_loss.backward()
    for left, right in zip(eager.parameters(), checkpointed.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.allclose(left.grad, right.grad, atol=_CUDA_FP32_GRAD_ATOL)


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


def test_float16_precision_falls_back_on_cpu() -> None:
    policy = resolve_precision("float16", "cpu")
    assert policy.resolved == "float32"
    assert policy.fallback_reason is not None
    assert not policy.needs_grad_scaler


def test_auto_precision_prefers_native_bf16_over_emulated_on_cuda(monkeypatch) -> None:
    """Simulates Turing: PyTorch's default (emulation-inclusive) BF16 check reports
    True, but the native-only check reports False. "auto" must prefer real FP16
    Tensor Core acceleration over BF16 that would silently run emulated."""

    def fake_is_bf16_supported(*, including_emulation: bool = True) -> bool:
        return including_emulation  # True by default; False for the native-only check

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", fake_is_bf16_supported)

    turing_auto = resolve_precision("auto", "cuda")
    assert turing_auto.resolved == "float16"
    assert turing_auto.needs_grad_scaler

    # An explicit bfloat16 request is still honored on "Turing" -- emulated, but
    # not silently redirected to FP16 behind the caller's back.
    explicit_bf16 = resolve_precision("bfloat16", "cuda")
    assert explicit_bf16.resolved == "bfloat16"
    assert not explicit_bf16.needs_grad_scaler

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_: True)
    ampere_auto = resolve_precision("auto", "cuda")
    assert ampere_auto.resolved == "bfloat16"
    assert not ampere_auto.needs_grad_scaler


# --- CUDA-only tests: MF-075 (FP16 + GradScaler) ------------------------------------------------
#
# Tolerance declared here, before running against this hardware, per the same "fixed before
# measurement" discipline as the MF-046/049 CUDA tests above. FP16 has more mantissa bits than
# BF16 (10 vs 7) but a narrower exponent range, so a comparable-or-tighter atol is expected --
# not assumed equal to BF16's without checking.
_CUDA_FP16_LOGIT_ATOL = 5e-2


@requires_cuda
def test_cuda_float16_full_and_cached_logits_match_within_declared_tolerance() -> None:
    torch.manual_seed(50)
    device = torch.device("cuda")
    config = ModelConfig.tiny_modern(max_seq_len=12, attention_impl="sdpa")
    model = MiniFrontier(config).to(device).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 8), device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        full = model(tokens).logits
        cache = KVCache.allocate(config, batch_size=1, device=device, capacity=8)
        pieces = [model(tokens[:, index : index + 1], cache=cache).logits for index in range(8)]
        cached = torch.cat(pieces, dim=1)
    assert cache.layers[0].dtype == torch.float16
    assert torch.isfinite(full).all() and torch.isfinite(cached).all()
    assert torch.allclose(full, cached, atol=_CUDA_FP16_LOGIT_ATOL)
    assert torch.equal(full.argmax(dim=-1), cached.argmax(dim=-1))


@requires_cuda
def test_cuda_float16_training_step_uses_gradscaler_and_stays_finite() -> None:
    torch.manual_seed(51)
    device = torch.device("cuda")
    model = MiniFrontier(ModelConfig.tiny_edu(n_layers=2, d_model=16, n_heads=2, d_ff=32)).to(
        device
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    config = TrainingConfig(
        max_updates=3,
        warmup_updates=0,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        precision="float16",
    )
    _, _, state, policy = train_updates(
        model, ListBatchProvider([TrainingBatch(tokens)]), config, device=device
    )
    assert policy.resolved == "float16"
    assert policy.needs_grad_scaler
    assert state.grad_scaler_state is not None
    assert math.isfinite(state.last_loss)
    # completed_updates advances unconditionally after scaler.step()/update(), whether or not the
    # scaler actually applied that particular step -- a skipped (inf/nan) step must not stall the
    # schedule, matching nanoGPT/nanochat's convention.
    assert state.completed_updates == 3
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )


@requires_cuda
def test_cuda_float16_checkpoint_resume_restores_gradscaler_state(tmp_path) -> None:
    torch.manual_seed(52)
    device = torch.device("cuda")
    model_config = ModelConfig.tiny_edu(n_layers=1, d_model=16, n_heads=2, d_ff=32, max_seq_len=8)
    interrupted = MiniFrontier(model_config).to(device)
    batches = [
        TrainingBatch(torch.tensor([[1, 2, 3, 4, 5, 6]])),
        TrainingBatch(torch.tensor([[6, 5, 4, 3, 2, 1]])),
    ]
    train_config = TrainingConfig(
        max_updates=6,
        warmup_updates=1,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        precision="float16",
    )
    optimizer, schedule, state, _ = train_updates(
        interrupted,
        ListBatchProvider(batches),
        train_config,
        device=device,
        stop_after_updates=3,
    )
    assert state.grad_scaler_state is not None
    saved_scale = state.grad_scaler_state["scale"]
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(
        checkpoint,
        interrupted,
        optimizer=optimizer,
        scheduler=schedule,
        trainer_state=state.to_dict(),
        data_cursor=ListBatchProvider(batches).state_dict(),
    )

    resumed = MiniFrontier(model_config).to(device)
    resumed_optimizer, _ = build_adamw(resumed, train_config)
    resumed_schedule = WarmupCosineSchedule(train_config)
    trainer_values, _ = load_training_checkpoint(
        checkpoint,
        resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_schedule,
        trusted_local_state=True,
    )
    resumed_state = TrainingState.from_dict(trainer_values)
    assert resumed_state.grad_scaler_state == state.grad_scaler_state
    _, _, final_state, _ = train_updates(
        resumed,
        ListBatchProvider(batches),
        train_config,
        device=device,
        optimizer=resumed_optimizer,
        schedule=resumed_schedule,
        state=resumed_state,
    )
    # The restored scale is the resumed run's starting point, not necessarily its
    # ending point -- GradScaler keeps adjusting as training continues. What must
    # hold is that resume did not silently reset to the scaler's default (65536.0).
    assert resumed_state.grad_scaler_state["scale"] == saved_scale
    assert math.isfinite(final_state.last_loss)
