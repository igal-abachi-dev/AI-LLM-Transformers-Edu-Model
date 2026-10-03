"""MF-168: tests for the from-scratch LoRA implementation."""

from __future__ import annotations

import torch
from torch import nn

from minifrontier.config import ModelConfig
from minifrontier.lora import (
    LoRALinear,
    apply_lora,
    lora_parameters,
    mark_only_lora_trainable,
    trainable_parameter_counts,
)
from minifrontier.model import MiniFrontier


def test_lora_linear_rejects_nonpositive_rank() -> None:
    base = nn.Linear(4, 4, bias=False)
    try:
        LoRALinear(base, rank=0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_lora_linear_rejects_a_biased_base_layer() -> None:
    base = nn.Linear(4, 4, bias=True)
    try:
        LoRALinear(base, rank=2)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_lora_linear_matches_the_base_layer_exactly_at_initialization() -> None:
    """The one detail that matters most: B starts at zero, so a freshly
    wrapped layer's forward pass must be byte-identical to the unwrapped
    base, not merely close."""

    torch.manual_seed(0)
    base = nn.Linear(16, 8, bias=False)
    x = torch.randn(3, 16)
    expected = base(x)

    wrapped = LoRALinear(base, rank=4)
    actual = wrapped(x)
    assert torch.equal(actual, expected)


def test_lora_linear_freezes_the_base_weight() -> None:
    base = nn.Linear(4, 4, bias=False)
    wrapped = LoRALinear(base, rank=2)
    assert wrapped.base.weight.requires_grad is False
    assert wrapped.lora_down.requires_grad is True
    assert wrapped.lora_up.requires_grad is True


def test_training_only_the_adapter_leaves_the_base_weight_unchanged_but_moves_the_output() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 4, bias=False)
    wrapped = LoRALinear(base, rank=2)
    original_base_weight = base.weight.clone()

    x = torch.randn(5, 8)
    target = torch.randn(5, 4)
    optimizer = torch.optim.SGD(wrapped.trainable_parameters(), lr=0.5)
    first_output = wrapped(x).detach().clone()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(wrapped(x), target)
        loss.backward()
        optimizer.step()

    # The frozen base weight never received a gradient and never moved.
    assert torch.equal(base.weight, original_base_weight)
    # The adapter alone genuinely changed what the layer computes.
    assert not torch.equal(wrapped(x).detach(), first_output)
    assert (
        torch.nn.functional.mse_loss(wrapped(x), target).item()
        < torch.nn.functional.mse_loss(first_output, target).item()
    )


def test_merged_weight_matches_a_real_forward_pass() -> None:
    """Real numerical correctness check, not just a shape check: a plain
    nn.Linear built from merged_weight() must reproduce the adapted layer's
    own output exactly."""

    torch.manual_seed(1)
    base = nn.Linear(6, 5, bias=False)
    wrapped = LoRALinear(base, rank=3)
    # Move the adapter away from its zero-init so merging is a real test.
    with torch.no_grad():
        wrapped.lora_up.copy_(torch.randn_like(wrapped.lora_up))

    x = torch.randn(4, 6)
    expected = wrapped(x)

    merged = nn.Linear(6, 5, bias=False)
    with torch.no_grad():
        merged.weight.copy_(wrapped.merged_weight())
    actual = merged(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_apply_lora_wraps_exactly_the_real_attention_projections() -> None:
    torch.manual_seed(0)
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=16))
    wrapped_names = apply_lora(model, target_pattern=r"\.(q_proj|k_proj|v_proj|out_proj)$", rank=2)

    # Real, specific names -- not just "something got wrapped".
    assert any(name.endswith("q_proj") for name in wrapped_names)
    assert any(name.endswith("k_proj") for name in wrapped_names)
    assert any(name.endswith("v_proj") for name in wrapped_names)
    assert any(name.endswith("out_proj") for name in wrapped_names)
    assert not any(name.endswith("gate_proj") for name in wrapped_names)
    assert len(wrapped_names) == 4 * model.config.n_layers

    for name in wrapped_names:
        assert isinstance(model.get_submodule(name), LoRALinear)


def test_apply_lora_does_not_touch_the_tied_lm_head_unless_targeted() -> None:
    torch.manual_seed(0)
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=16))
    apply_lora(model, target_pattern=r"\.(q_proj|k_proj|v_proj|out_proj)$", rank=2)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, LoRALinear)
    assert model.lm_head.weight.requires_grad  # untouched, still trainable


def test_lora_parameters_matches_the_real_expected_count() -> None:
    torch.manual_seed(0)
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=16))
    rank = 2
    wrapped_names = apply_lora(
        model, target_pattern=r"\.(q_proj|k_proj|v_proj|out_proj)$", rank=rank
    )
    params = lora_parameters(model)
    assert len(params) == 2 * len(wrapped_names)  # one down + one up per wrapped layer

    expected_elements = sum(p.numel() for p in params)
    # Real, hand-computed expectation from each wrapped layer's own real shape.
    hand_computed = 0
    for name in wrapped_names:
        layer = model.get_submodule(name)
        assert isinstance(layer, LoRALinear)
        hand_computed += rank * layer.base.in_features + layer.base.out_features * rank
    assert expected_elements == hand_computed


def test_mark_only_lora_trainable_freezes_everything_else() -> None:
    torch.manual_seed(0)
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=16))
    apply_lora(model, target_pattern=r"\.(q_proj|k_proj|v_proj|out_proj)$", rank=2)
    mark_only_lora_trainable(model)

    lora_param_ids = {id(p) for p in lora_parameters(model)}
    for parameter in model.parameters():
        if id(parameter) in lora_param_ids:
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad


def test_trainable_parameter_counts_shows_the_real_reduction() -> None:
    """The whole real point of LoRA, as a concrete number: trainable
    parameters must be a small fraction of the model's real total once only
    the adapters are left trainable."""

    torch.manual_seed(0)
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=16))
    apply_lora(model, target_pattern=r"\.(q_proj|k_proj|v_proj|out_proj)$", rank=2)
    mark_only_lora_trainable(model)

    trainable, total = trainable_parameter_counts(model)
    assert 0 < trainable < total
    assert trainable / total < 0.5  # real, substantial reduction even at this tiny scale


def test_apply_lora_rejects_nonpositive_rank() -> None:
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=16))
    try:
        apply_lora(model, target_pattern=r"q_proj", rank=0)
        raised = False
    except ValueError:
        raised = True
    assert raised
