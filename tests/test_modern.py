from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

import minifrontier.attention as attention_module
from minifrontier.attention import (
    CausalSelfAttention,
    block_mask_cache_size,
    clear_block_mask_cache,
    flex_attention_compilation_enabled,
    set_flex_attention_compilation,
)
from minifrontier.cache import KVCache
from minifrontier.config import ModelConfig
from minifrontier.masking import build_attention_mask
from minifrontier.model import MiniFrontier
from minifrontier.rope import RoPE


def test_gqa_manual_and_sdpa_output_and_gradient_parity() -> None:
    torch.manual_seed(20)
    config = ModelConfig.tiny_modern(n_layers=4, attention_impl="sdpa")
    manual_model = MiniFrontier(config)
    sdpa_model = MiniFrontier(config)
    sdpa_model.load_state_dict(manual_model.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    manual = manual_model(tokens, labels=tokens, attention_impl="manual")
    sdpa = sdpa_model(tokens, labels=tokens, attention_impl="sdpa")
    assert torch.allclose(manual.logits, sdpa.logits, atol=1e-5)
    assert manual.loss is not None and sdpa.loss is not None
    manual.loss.backward()
    sdpa.loss.backward()
    for left, right in zip(manual_model.parameters(), sdpa_model.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.allclose(left.grad, right.grad, atol=2e-5)


def test_sdpa_receives_compact_kv_and_native_gqa(monkeypatch) -> None:
    config = ModelConfig.tiny_modern(n_layers=4, attention_impl="sdpa")
    attention = CausalSelfAttention(config, layer_index=3).eval()
    inputs = torch.randn(1, 5, config.d_model)
    rope = RoPE(config.head_dim, config.max_seq_len)
    cosine, sine = rope(torch.arange(5), dtype=inputs.dtype, device=inputs.device)
    original = F.scaled_dot_product_attention
    observed: dict[str, object] = {}

    def recording_sdpa(query, key, value, **kwargs):
        observed["query_heads"] = query.shape[1]
        observed["key_heads"] = key.shape[1]
        observed.update(kwargs)
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    attention(inputs, cosine, sine)
    assert observed["query_heads"] == config.n_heads
    assert observed["key_heads"] == config.n_kv_heads
    assert observed["enable_gqa"] is True
    assert observed["attn_mask"] is None
    assert observed["is_causal"] is True


def test_qk_norm_runs_before_rope(monkeypatch) -> None:
    config = ModelConfig.tiny_modern(n_layers=4, attention_impl="sdpa")
    attention = CausalSelfAttention(config, layer_index=3).eval()
    inputs = torch.randn(2, 5, config.d_model) * 17
    rope = RoPE(config.head_dim, config.max_seq_len)
    cosine, sine = rope(torch.arange(5), dtype=inputs.dtype, device=inputs.device)
    original = attention_module.apply_rotary
    observed_rms: list[torch.Tensor] = []

    def recording_rotary(tensor, cos, sin):
        observed_rms.append(tensor.float().pow(2).mean(dim=-1).sqrt())
        return original(tensor, cos, sin)

    monkeypatch.setattr(attention_module, "apply_rotary", recording_rotary)
    attention(inputs, cosine, sine)
    assert len(observed_rms) == 2
    for rms in observed_rms:
        assert torch.allclose(rms, torch.ones_like(rms), atol=2e-4)


def test_global_nope_skips_only_global_rope() -> None:
    torch.manual_seed(21)
    config = ModelConfig.tiny_modern(global_position_encoding="none", attention_impl="sdpa")
    inputs = torch.randn(1, 6, config.d_model)
    mask = build_attention_mask(6, 6, window_size=config.local_window)
    rope = RoPE(config.head_dim, config.max_seq_len)
    cos_a, sin_a = rope(torch.arange(6), dtype=inputs.dtype, device=inputs.device)
    cos_b = torch.ones_like(cos_a)
    sin_b = torch.zeros_like(sin_a)

    local = CausalSelfAttention(config, layer_index=0).eval()
    global_attention = CausalSelfAttention(config, layer_index=3).eval()
    assert not torch.allclose(
        local(inputs, cos_a, sin_a, attention_mask=mask),
        local(inputs, cos_b, sin_b, attention_mask=mask),
    )
    assert torch.equal(
        global_attention(inputs, cos_a, sin_a),
        global_attention(inputs, cos_b, sin_b),
    )


def test_flex_local_gqa_matches_manual_and_reuses_block_mask() -> None:
    torch.manual_seed(22)
    clear_block_mask_cache()
    config = ModelConfig.tiny_modern(n_layers=4, attention_impl="auto")
    attention = CausalSelfAttention(config, layer_index=0).eval()
    inputs = torch.randn(1, 6, config.d_model)
    rope = RoPE(config.head_dim, config.max_seq_len)
    cosine, sine = rope(torch.arange(6), dtype=inputs.dtype, device=inputs.device)
    mask = build_attention_mask(6, 6, window_size=config.local_window)
    manual = attention(
        inputs,
        cosine,
        sine,
        implementation="manual",
        attention_mask=mask,
    )
    with torch.no_grad():
        flex = attention(inputs, cosine, sine, implementation="flex")
    assert torch.allclose(manual, flex, atol=3e-5)
    assert block_mask_cache_size() == 1
    with torch.no_grad():
        attention(inputs, cosine, sine, implementation="flex")
    assert block_mask_cache_size() == 1


def test_flex_block_mask_cache_evicts_oldest_entry_once_full(monkeypatch) -> None:
    """A long-running process seeing varied lengths must not grow this cache forever."""

    clear_block_mask_cache()
    monkeypatch.setattr(attention_module, "_BLOCK_MASK_CACHE_MAX_SIZE", 2)
    config = ModelConfig.tiny_modern(n_layers=1, attention_impl="auto")
    attention = CausalSelfAttention(config, layer_index=0).eval()
    rope = RoPE(config.head_dim, config.max_seq_len)
    for length in (4, 5, 6):
        inputs = torch.randn(1, length, config.d_model)
        cosine, sine = rope(torch.arange(length), dtype=inputs.dtype, device=inputs.device)
        with torch.no_grad():
            attention(inputs, cosine, sine, implementation="flex")
    assert block_mask_cache_size() == 2


def test_flex_block_mask_built_under_inference_mode_is_never_cached() -> None:
    """Regression, reproduced for real during MF-087's tokenizer comparison:
    evaluating a reused checkpoint (torch.inference_mode()) before training a
    second, same-shape model in the same process crashed with "Inference
    tensors cannot be saved for backward" -- the shared block-mask cache does
    not know a later lookup with the same shape key will be a training
    (autograd-tracked) call rather than another inference-mode one, so an
    inference-mode-built mask must never be shared at all. CPU-safe half of
    the regression: confirms nothing gets cached (no backward pass needed,
    since FlexAttention doesn't support backward on CPU at all -- see the
    CUDA-gated full reproduction below for the actual crash scenario)."""

    clear_block_mask_cache()
    config = ModelConfig.tiny_modern(n_layers=1, attention_impl="auto")
    rope = RoPE(config.head_dim, config.max_seq_len)
    length = 6

    attention = CausalSelfAttention(config, layer_index=0).eval()
    with torch.inference_mode():
        inputs = torch.randn(1, length, config.d_model)
        cosine, sine = rope(torch.arange(length), dtype=inputs.dtype, device=inputs.device)
        attention(inputs, cosine, sine, implementation="flex")
    assert block_mask_cache_size() == 0


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@requires_cuda
def test_cuda_flex_block_mask_from_inference_mode_does_not_break_later_training() -> None:
    """Full reproduction of the real crash: an inference_mode()-wrapped forward
    pass (e.g. evaluating a reused checkpoint) followed by a same-shape
    training forward+backward in the same process, on the device where the
    bug actually occurred."""

    clear_block_mask_cache()
    device = torch.device("cuda")
    config = ModelConfig.tiny_modern(n_layers=1, attention_impl="auto")
    rope = RoPE(config.head_dim, config.max_seq_len)
    length = 6

    inference_attention = CausalSelfAttention(config, layer_index=0).eval().to(device)
    with torch.inference_mode():
        inference_inputs = torch.randn(1, length, config.d_model, device=device)
        cosine, sine = rope(
            torch.arange(length, device=device),
            dtype=inference_inputs.dtype,
            device=device,
        )
        inference_attention(inference_inputs, cosine, sine, implementation="flex")
    assert block_mask_cache_size() == 0

    training_attention = CausalSelfAttention(config, layer_index=0).to(device)
    training_inputs = torch.randn(1, length, config.d_model, device=device, requires_grad=True)
    cosine, sine = rope(
        torch.arange(length, device=device), dtype=training_inputs.dtype, device=device
    )
    output = training_attention(training_inputs, cosine, sine, implementation="flex")
    output.sum().backward()  # would raise "Inference tensors cannot be saved for backward" pre-fix
    assert training_inputs.grad is not None
    assert block_mask_cache_size() == 1


@pytest.mark.slow
def test_compiled_flex_attention_matches_eager_flex() -> None:
    """Compiling FlexAttention directly (not just the outer model) must not change the answer."""

    torch.manual_seed(24)
    clear_block_mask_cache()
    config = ModelConfig.tiny_modern(n_layers=4, attention_impl="auto")
    attention = CausalSelfAttention(config, layer_index=0).eval()
    inputs = torch.randn(1, 6, config.d_model)
    rope = RoPE(config.head_dim, config.max_seq_len)
    cosine, sine = rope(torch.arange(6), dtype=inputs.dtype, device=inputs.device)
    with torch.no_grad():
        eager_flex = attention(inputs, cosine, sine, implementation="flex")

    assert not flex_attention_compilation_enabled()
    set_flex_attention_compilation(True)
    try:
        assert flex_attention_compilation_enabled()
        with torch.no_grad():
            compiled_flex = attention(inputs, cosine, sine, implementation="flex")
        # A second call must reuse the same compiled callable rather than
        # recompiling (the shape has not changed).
        with torch.no_grad():
            compiled_flex_again = attention(inputs, cosine, sine, implementation="flex")
    finally:
        set_flex_attention_compilation(False)

    assert not flex_attention_compilation_enabled()
    assert torch.allclose(eager_flex, compiled_flex, atol=3e-5)
    assert torch.equal(compiled_flex, compiled_flex_again)


def test_flex_attention_compilation_failure_falls_back_and_stays_disabled() -> None:
    """A failed compiled call must degrade to eager and never retry the broken path.

    This stands in the already-compiled callable directly (bypassing `torch.compile`
    itself) rather than breaking `torch.compile` globally: FlexAttention's own eager
    fallback also relies on `torch.compile` internally for its mask_mod handling, so
    breaking `torch.compile` system-wide would take down the correctness-preserving
    fallback path too and not exercise what this test actually cares about --that a
    compiled *kernel* failure degrades safely and does not retry every call.
    """

    torch.manual_seed(25)
    clear_block_mask_cache()
    config = ModelConfig.tiny_modern(n_layers=4, attention_impl="auto")
    attention = CausalSelfAttention(config, layer_index=0).eval()
    inputs = torch.randn(1, 6, config.d_model)
    rope = RoPE(config.head_dim, config.max_seq_len)
    cosine, sine = rope(torch.arange(6), dtype=inputs.dtype, device=inputs.device)
    with torch.no_grad():
        eager_flex = attention(inputs, cosine, sine, implementation="flex")

    calls = {"count": 0}

    def broken_compiled(*_args, **_kwargs):
        calls["count"] += 1
        raise RuntimeError("simulated compiled-kernel failure")

    set_flex_attention_compilation(True)
    attention_module._compiled_flex_attention = broken_compiled
    try:
        with pytest.warns(RuntimeWarning, match="falling back to eager"), torch.no_grad():
            first = attention(inputs, cosine, sine, implementation="flex")
        assert calls["count"] == 1
        # A second call must skip the already-failed compiled path entirely --
        # no retry, no second warning.
        with torch.no_grad():
            second = attention(inputs, cosine, sine, implementation="flex")
        assert calls["count"] == 1
    finally:
        set_flex_attention_compilation(False)
    assert torch.equal(eager_flex, first)
    assert torch.equal(eager_flex, second)


@pytest.mark.parametrize("global_position_encoding", ["rope", "none"])
def test_modern_cached_full_and_chunked_logits_match(global_position_encoding: str) -> None:
    torch.manual_seed(23)
    config = ModelConfig.tiny_modern(
        max_seq_len=16,
        local_window=4,
        global_position_encoding=global_position_encoding,  # type: ignore[arg-type]
        attention_impl="sdpa",
    )
    model = MiniFrontier(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 11))
    full = model(tokens).logits
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        dtype=model.token_embedding.weight.dtype,
        capacity=11,
    )
    cached = torch.cat(
        (
            model(tokens[:, :3], cache=cache).logits,
            model(tokens[:, 3:7], cache=cache).logits,
            model(tokens[:, 7:], cache=cache).logits,
        ),
        dim=1,
    )
    assert torch.allclose(full, cached, atol=2e-5)
    assert torch.equal(full.argmax(dim=-1), cached.argmax(dim=-1))


def test_modern_gqa_reduces_parameters_and_cache_bytes() -> None:
    edu_config = ModelConfig.tiny_edu(n_layers=4, d_model=32, n_heads=4, d_ff=96)
    modern_config = ModelConfig.tiny_modern(
        n_layers=4,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=96,
        attention_impl="sdpa",
    )
    edu = MiniFrontier(edu_config)
    modern = MiniFrontier(modern_config)
    assert modern.parameter_count() < edu.parameter_count()
    edu_cache = KVCache.allocate(
        edu_config, batch_size=1, device="cpu", dtype=torch.float32, capacity=16
    )
    modern_cache = KVCache.allocate(
        modern_config, batch_size=1, device="cpu", dtype=torch.float32, capacity=16
    )
    assert modern_cache.allocated_bytes() == edu_cache.allocated_bytes() // 2


def test_auto_dispatches_local_flex_and_global_sdpa() -> None:
    config = ModelConfig.tiny_modern(attention_impl="auto")
    assert config.attention_impl_for_layer(0) == "flex"
    assert config.attention_impl_for_layer(3) == "sdpa"
    assert config.attention_impl_for_layer(0, "manual") == "manual"


def test_auto_local_single_token_cache_decode_dispatches_to_sdpa() -> None:
    config = ModelConfig.tiny_modern(attention_impl="auto")
    attention = CausalSelfAttention(config, layer_index=0)
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        capacity=12,
        bounded_local=True,
    )
    assert (
        attention.resolved_implementation(
            cache=cache.layers[0],
            sequence_length=1,
        )
        == "sdpa"
    )


def test_auto_generation_does_not_grow_flex_mask_cache_per_decode_token() -> None:
    clear_block_mask_cache()
    model = MiniFrontier(ModelConfig.tiny_modern(attention_impl="auto")).eval()
    model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4)
    assert block_mask_cache_size() == 1


def test_modern_cache_never_reads_unwritten_slots_and_replays_after_truncate() -> None:
    torch.manual_seed(24)
    config = ModelConfig.tiny_modern(
        max_seq_len=12,
        local_window=4,
        attention_impl="sdpa",
    )
    model = MiniFrontier(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 7))

    reference_cache = KVCache.allocate(config, batch_size=1, device="cpu", capacity=12)
    model(tokens[:, :3], cache=reference_cache)
    reference = model(tokens[:, 3:4], cache=reference_cache).logits

    poisoned_cache = KVCache.allocate(config, batch_size=1, device="cpu", capacity=12)
    model(tokens[:, :3], cache=poisoned_cache)
    for layer in poisoned_cache.layers:
        assert layer.keys is not None and layer.values is not None
        layer.keys[:, :, 3:].fill_(1e6)
        layer.values[:, :, 3:].fill_(-1e6)
    actual = model(tokens[:, 3:4], cache=poisoned_cache).logits
    assert torch.equal(reference, actual)

    poisoned_cache.truncate(3)
    replay = model(tokens[:, 3:7], cache=poisoned_cache).logits
    clean_cache = KVCache.allocate(config, batch_size=1, device="cpu", capacity=12)
    model(tokens[:, :3], cache=clean_cache)
    expected = model(tokens[:, 3:7], cache=clean_cache).logits
    assert torch.equal(replay, expected)
