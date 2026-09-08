from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minifrontier.cache import KVCache
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier

ROOT = Path(__file__).parents[1]


def test_model_forward_loss_backward_and_tied_weights() -> None:
    torch.manual_seed(6)
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    tokens = torch.randint(0, config.vocab_size, (2, 12))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 12, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.lm_head.weight is model.token_embedding.weight
    output.loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_model_manual_and_sdpa_logits_match() -> None:
    torch.manual_seed(7)
    model = MiniFrontier(ModelConfig.tiny_edu()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    manual = model(tokens, attention_impl="manual").logits
    sdpa = model(tokens, attention_impl="sdpa").logits
    assert torch.allclose(manual, sdpa, atol=5e-6)


def test_model_logits_are_causal() -> None:
    torch.manual_seed(8)
    model = MiniFrontier(ModelConfig.tiny_edu()).eval()
    first = torch.randint(0, model.config.vocab_size, (1, 10))
    second = first.clone()
    second[:, -1] = (second[:, -1] + 1) % model.config.vocab_size
    assert torch.allclose(model(first).logits[:, :-1], model(second).logits[:, :-1], atol=1e-5)


def test_generation_is_deterministic_and_restores_mode() -> None:
    model = MiniFrontier(ModelConfig.tiny_edu()).train()
    prompt = torch.tensor([[1, 2, 3]])
    first = model.generate(prompt, max_new_tokens=3)
    second = model.generate(prompt, max_new_tokens=3)
    assert torch.equal(first, second)
    assert first.shape == (1, 6)
    assert model.training


def test_depth_scaled_residual_initialization() -> None:
    torch.manual_seed(11)
    config = ModelConfig.tiny_edu(n_layers=4, d_model=64, n_heads=4, d_ff=256)
    model = MiniFrontier(config)
    base_expected = config.d_model**-0.5
    residual_expected = base_expected / (2 * config.n_layers) ** 0.5
    assert model.blocks[0].attention.q_proj.weight.std().item() == pytest.approx(
        base_expected, rel=0.08
    )
    assert model.blocks[0].attention.out_proj.weight.std().item() == pytest.approx(
        residual_expected, rel=0.08
    )
    assert model.blocks[0].feed_forward.down_proj.weight.std().item() == pytest.approx(
        residual_expected, rel=0.08
    )


def test_residual_std_damping_can_be_disabled_for_the_layer_norm_scaling_ablation() -> None:
    torch.manual_seed(30)
    config = ModelConfig.tiny_edu(n_layers=4, d_model=64, n_heads=4, d_ff=256)
    config = replace(config, residual_std_damping=False)
    model = MiniFrontier(config)
    base_expected = config.d_model**-0.5
    # Without the depth-scaled init damping, out_proj/down_proj get the same
    # plain init as every other Linear -- no 1/sqrt(2*n_layers) shrink.
    assert model.blocks[0].attention.out_proj.weight.std().item() == pytest.approx(
        base_expected, rel=0.08
    )
    assert model.blocks[0].feed_forward.down_proj.weight.std().item() == pytest.approx(
        base_expected, rel=0.08
    )


def test_layer_norm_scaling_disabled_by_default() -> None:
    config = ModelConfig.tiny_edu(n_layers=3)
    model = MiniFrontier(config)
    assert all(block.layer_norm_scale is None for block in model.blocks)


def test_layer_norm_scaling_uses_one_indexed_inverse_sqrt_depth() -> None:
    # Modern-only (2026-09-08 decision): Edu stays the classic architecture.
    config = ModelConfig.tiny_modern(n_layers=4, attention_impl="sdpa")
    config = replace(config, layer_norm_scaling=True)
    model = MiniFrontier(config)
    for layer_index, block in enumerate(model.blocks):
        assert block.layer_norm_scale == pytest.approx((layer_index + 1) ** -0.5)


def test_layer_norm_scaling_changes_forward_output() -> None:
    # Modern-only (2026-09-08 decision): Edu stays the classic architecture.
    torch.manual_seed(31)
    config = ModelConfig.tiny_modern(
        n_layers=4, d_model=64, n_heads=4, n_kv_heads=2, d_ff=256, attention_impl="sdpa"
    )
    baseline = MiniFrontier(config).eval()
    torch.manual_seed(31)
    scaled = MiniFrontier(replace(config, layer_norm_scaling=True)).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 9))
    assert not torch.allclose(baseline(tokens).logits, scaled(tokens).logits)


def test_value_residual_disabled_has_no_gates() -> None:
    config = ModelConfig.tiny_modern(n_layers=4)
    model = MiniFrontier(config)
    assert all(block.attention.value_residual_gate is None for block in model.blocks)


def test_value_residual_first_layer_never_gets_a_gate() -> None:
    config = replace(ModelConfig.tiny_modern(n_layers=4), value_residual=True)
    model = MiniFrontier(config)
    assert model.blocks[0].attention.value_residual_gate is None
    assert all(block.attention.value_residual_gate is not None for block in model.blocks[1:])


def test_value_residual_zero_initialized_gate_matches_disabled_forward() -> None:
    """The zero-init claim: enabling the flag must not change forward output
    until a gate actually learns something away from zero."""

    torch.manual_seed(32)
    config = ModelConfig.tiny_modern(
        n_layers=4, d_model=32, n_heads=4, n_kv_heads=2, d_ff=96, attention_impl="sdpa"
    )
    baseline = MiniFrontier(config).eval()
    torch.manual_seed(32)
    gated = MiniFrontier(replace(config, value_residual=True)).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 9))
    assert torch.equal(baseline(tokens).logits, gated(tokens).logits)


def test_value_residual_nonzero_gate_changes_forward_output() -> None:
    torch.manual_seed(33)
    config = replace(
        ModelConfig.tiny_modern(
            n_layers=4, d_model=32, n_heads=4, n_kv_heads=2, d_ff=96, attention_impl="sdpa"
        ),
        value_residual=True,
    )
    model = MiniFrontier(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 9))
    zero_gate_logits = model(tokens).logits
    with torch.no_grad():
        model.blocks[2].attention.value_residual_gate.fill_(1.0)
    nonzero_gate_logits = model(tokens).logits
    assert not torch.allclose(zero_gate_logits, nonzero_gate_logits)


def test_value_residual_cached_and_uncached_logits_match() -> None:
    torch.manual_seed(34)
    config = replace(
        ModelConfig.tiny_modern(
            max_seq_len=16,
            local_window=4,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=96,
            attention_impl="sdpa",
        ),
        value_residual=True,
    )
    model = MiniFrontier(config).eval()
    with torch.no_grad():
        for block in model.blocks[1:]:
            block.attention.value_residual_gate.fill_(0.5)
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


def test_hidden_states_are_none_by_default_and_populated_when_requested() -> None:
    torch.manual_seed(9)
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    default_output = model(tokens, labels=tokens)
    assert default_output.hidden_states is None

    with_hidden = model(tokens, labels=tokens, return_hidden_states=True)
    assert with_hidden.hidden_states is not None
    assert with_hidden.hidden_states.shape == (2, 7, config.d_model)
    # Requesting hidden states must not change the logits/loss actually returned.
    assert torch.allclose(with_hidden.logits, default_output.logits)
    assert torch.allclose(with_hidden.loss, default_output.loss)


def test_skip_logits_returns_only_hidden_states() -> None:
    torch.manual_seed(10)
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    full_output = model(tokens, return_hidden_states=True)
    skipped_output = model(tokens, return_hidden_states=True, skip_logits=True)

    assert skipped_output.logits is None
    assert skipped_output.hidden_states is not None
    assert torch.equal(skipped_output.hidden_states, full_output.hidden_states)


def test_skip_logits_requires_return_hidden_states_and_rejects_labels_or_logits_to_keep() -> None:
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    with pytest.raises(ValueError, match="return_hidden_states"):
        model(tokens, skip_logits=True)
    with pytest.raises(ValueError, match="skip_logits"):
        model(tokens, labels=tokens, skip_logits=True, return_hidden_states=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        model(tokens, skip_logits=True, return_hidden_states=True, logits_to_keep=1)


def test_model_rejects_long_or_non_integer_tokens() -> None:
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=8))
    with pytest.raises(ValueError, match="sequence length"):
        model(torch.zeros(1, 9, dtype=torch.long))
    with pytest.raises(ValueError, match="integer"):
        model(torch.zeros(1, 3))


@pytest.mark.slow
def test_50m_edu_instantiates_at_frozen_parameter_count() -> None:
    config = ModelConfig.from_toml(ROOT / "configs" / "50m-edu.toml")
    model = MiniFrontier(config)
    assert model.parameter_count() == 53_361_152
