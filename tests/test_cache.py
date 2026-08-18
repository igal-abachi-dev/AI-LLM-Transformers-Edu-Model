import pytest
import torch

from minifrontier.cache import KVCache, LayerKVCache
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def test_layer_cache_appends_without_reallocation_and_resets() -> None:
    cache = LayerKVCache.allocate(
        batch_size=2,
        n_kv_heads=4,
        capacity=8,
        head_dim=6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    pointer = cache.keys.data_ptr()
    first = torch.randn(2, 4, 3, 6)
    keys, values = cache.append(first, first + 1, start_pos=0)
    assert keys.shape == values.shape == (2, 4, 3, 6)
    second = torch.randn(2, 4, 2, 6)
    keys, _ = cache.append(second, second + 1, start_pos=3)
    assert keys.shape == (2, 4, 5, 6)
    assert cache.keys.data_ptr() == pointer
    assert cache.length == 5
    cache.reset()
    assert cache.length == 0


def test_layer_cache_rejects_position_shape_dtype_and_overflow() -> None:
    cache = LayerKVCache.allocate(
        batch_size=1,
        n_kv_heads=2,
        capacity=3,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    update = torch.randn(1, 2, 2, 4)
    with pytest.raises(ValueError, match="start_pos"):
        cache.append(update, update, start_pos=1)
    with pytest.raises(ValueError, match="dtype"):
        cache.append(update.double(), update.double(), start_pos=0)
    with pytest.raises(ValueError, match="capacity"):
        cache.append(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4), start_pos=0)


def test_full_and_token_by_token_cached_logits_match() -> None:
    torch.manual_seed(12)
    config = ModelConfig.tiny_edu(max_seq_len=16)
    model = MiniFrontier(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 10))
    full = model(tokens).logits
    cache = KVCache.allocate(
        config,
        batch_size=2,
        device="cpu",
        dtype=model.token_embedding.weight.dtype,
        capacity=10,
    )
    pieces = [model(tokens[:, index : index + 1], cache=cache).logits for index in range(10)]
    cached = torch.cat(pieces, dim=1)
    assert torch.allclose(full, cached, atol=1e-5)
    assert torch.equal(full.argmax(dim=-1), cached.argmax(dim=-1))
    assert cache.length == 10


def test_chunked_cached_logits_use_offset_causal_mask() -> None:
    torch.manual_seed(13)
    config = ModelConfig.tiny_edu(max_seq_len=16)
    model = MiniFrontier(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 9))
    full = model(tokens).logits
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        dtype=model.token_embedding.weight.dtype,
        capacity=9,
    )
    first = model(tokens[:, :3], cache=cache).logits
    second = model(tokens[:, 3:], cache=cache).logits
    cached = torch.cat((first, second), dim=1)
    assert torch.allclose(cached, full, atol=1e-5)
    assert torch.equal(cached.argmax(dim=-1), full.argmax(dim=-1))


def test_cache_is_inference_only_and_rolls_back_failed_append() -> None:
    config = ModelConfig.tiny_edu(max_seq_len=8)
    model = MiniFrontier(config)
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        dtype=model.token_embedding.weight.dtype,
        capacity=2,
    )
    with pytest.raises(ValueError, match="inference-only"):
        model(torch.ones(1, 1, dtype=torch.long), cache=cache)
    model.eval()
    with pytest.raises(ValueError, match="capacity"):
        model(torch.ones(1, 3, dtype=torch.long), cache=cache)
    assert cache.length == 0


def test_bounded_local_ring_cache_matches_full_history_across_wrap_and_chunks() -> None:
    torch.manual_seed(25)
    config = ModelConfig.tiny_modern(
        max_seq_len=16,
        local_window=4,
        attention_impl="sdpa",
    )
    model = MiniFrontier(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 12))
    full = model(tokens).logits
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        capacity=12,
        bounded_local=True,
    )
    cached = torch.cat(
        (
            model(tokens[:, :3], cache=cache).logits,
            model(tokens[:, 3:7], cache=cache).logits,
            model(tokens[:, 7:9], cache=cache).logits,
            model(tokens[:, 9:], cache=cache).logits,
        ),
        dim=1,
    )
    assert torch.allclose(full, cached, atol=2e-5)
    assert torch.equal(full.argmax(dim=-1), cached.argmax(dim=-1))
    assert [layer.capacity for layer in cache.layers] == [4, 4, 4, 12]
    assert [layer.ring for layer in cache.layers] == [True, True, True, False]
    reference = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        capacity=12,
    )
    assert cache.allocated_bytes() < reference.allocated_bytes()
    cache.reset()
    assert cache.length == 0
    assert all(layer.visible_start == 0 for layer in cache.layers)


def test_bounded_local_ring_single_token_decode_matches_after_multiple_wraps() -> None:
    """The mask-free optimized decode view is current + at most W-1 history entries."""

    torch.manual_seed(123)
    config = ModelConfig.tiny_modern(
        max_seq_len=16,
        local_window=4,
        attention_impl="sdpa",
    )
    model = MiniFrontier(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 12))
    full = model(tokens).logits
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        capacity=12,
        bounded_local=True,
    )
    cached = torch.cat(
        [model(tokens[:, index : index + 1], cache=cache).logits for index in range(12)],
        dim=1,
    )
    assert torch.allclose(full, cached, atol=2e-5)
    assert torch.equal(full.argmax(dim=-1), cached.argmax(dim=-1))


def test_ring_cache_rolls_back_overwritten_slots_after_failed_forward(monkeypatch) -> None:
    torch.manual_seed(26)
    config = ModelConfig.tiny_modern(
        max_seq_len=12,
        local_window=4,
        attention_impl="sdpa",
    )
    model = MiniFrontier(config).eval()
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        capacity=12,
        bounded_local=True,
    )
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    model(tokens[:, :6], cache=cache)
    local = cache.layers[0]
    assert local.keys is not None and local.values is not None
    keys_before = local.keys.clone()
    values_before = local.values.clone()

    with monkeypatch.context() as patch:

        def fail(*_args, **_kwargs):
            raise RuntimeError("injected failure")

        patch.setattr(model.blocks[3].attention, "forward", fail)
        with pytest.raises(RuntimeError, match="injected"):
            model(tokens[:, 6:7], cache=cache)

    assert cache.length == 6
    assert torch.equal(local.keys, keys_before)
    assert torch.equal(local.values, values_before)
    replay = model(tokens[:, 6:], cache=cache).logits
    clean = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        capacity=12,
        bounded_local=True,
    )
    model(tokens[:, :6], cache=clean)
    expected = model(tokens[:, 6:], cache=clean).logits
    assert torch.equal(replay, expected)


def test_ring_cache_rejects_branch_truncation_after_committed_wrap() -> None:
    config = ModelConfig.tiny_modern(max_seq_len=12, local_window=4, attention_impl="sdpa")
    model = MiniFrontier(config).eval()
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        capacity=12,
        bounded_local=True,
    )
    model(torch.arange(6).unsqueeze(0), cache=cache)
    with pytest.raises(ValueError, match="committed history"):
        cache.truncate(5)
