from pathlib import Path

import pytest
import torch

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
