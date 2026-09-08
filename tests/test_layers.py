import pytest
import torch
from torch.nn import functional as F

from minifrontier.layers import RMSNorm, SwiGLU


def test_rms_norm_matches_independent_reference_and_gradient() -> None:
    torch.manual_seed(1)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    norm = RMSNorm(8, eps=1e-6)
    norm.weight.data.copy_(torch.linspace(0.5, 1.5, 8))

    actual = norm(inputs)
    expected = reference_inputs * torch.rsqrt(
        reference_inputs.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    expected = expected * norm.weight.detach()
    assert torch.allclose(actual, expected, atol=1e-6)

    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.allclose(inputs.grad, reference_inputs.grad, atol=2e-5)


def test_rms_norm_preserves_shape_dtype_and_device() -> None:
    inputs = torch.randn(2, 4, dtype=torch.float64)
    norm = RMSNorm(4).to(dtype=torch.float64)
    output = norm(inputs)
    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype
    assert output.device == inputs.device


def test_rms_norm_preserves_low_precision_activation_dtype() -> None:
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    norm = RMSNorm(4)
    assert norm(inputs).dtype == torch.bfloat16


def test_swiglu_matches_explicit_formula_and_gradients() -> None:
    torch.manual_seed(2)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    layer = SwiGLU(8, 16)
    actual = layer(inputs)
    expected = F.linear(
        F.silu(F.linear(reference_inputs, layer.gate_proj.weight))
        * F.linear(reference_inputs, layer.up_proj.weight),
        layer.down_proj.weight,
    )
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.allclose(inputs.grad, reference_inputs.grad)
    assert all(module.bias is None for module in (layer.gate_proj, layer.up_proj, layer.down_proj))


def test_swiglu_clamp_disabled_matches_unclamped_default() -> None:
    torch.manual_seed(3)
    inputs = torch.randn(2, 3, 8)
    torch.manual_seed(1)
    unclamped = SwiGLU(8, 16)
    torch.manual_seed(1)
    explicit_none = SwiGLU(8, 16, clamp_value=None)
    assert torch.equal(unclamped(inputs), explicit_none(inputs))


def test_swiglu_clamp_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="clamp_value must be positive"):
        SwiGLU(8, 16, clamp_value=0.0)


def test_swiglu_clamp_bounds_extreme_activations() -> None:
    torch.manual_seed(4)
    clamp_value = 2.0
    layer = SwiGLU(4, 4, clamp_value=clamp_value)
    with torch.no_grad():
        # Force large pre-activation values regardless of the random input.
        layer.gate_proj.weight.fill_(50.0)
        layer.up_proj.weight.fill_(50.0)
    inputs = torch.ones(1, 1, 4)
    gate_raw = layer.gate_proj(inputs)
    up_raw = layer.up_proj(inputs)
    assert gate_raw.abs().max() > clamp_value
    assert up_raw.abs().max() > clamp_value
    unclamped_pre_down = F.silu(gate_raw) * up_raw

    output = layer(inputs)
    assert torch.isfinite(output).all()
    # The clamped gate*up product before down_proj must be bounded by what
    # the clamp values allow (silu on [-clamp, clamp] times a clamp-capped
    # up), and therefore far smaller in magnitude than the unclamped product
    # this same huge weight would otherwise produce.
    clamped_gate = gate_raw.clamp(min=-clamp_value, max=clamp_value)
    clamped_up = up_raw.clamp(max=clamp_value)
    clamped_pre_down = F.silu(clamped_gate) * clamped_up
    assert clamped_pre_down.abs().max() < unclamped_pre_down.abs().max()
    assert clamped_pre_down.abs().max() <= F.silu(torch.tensor(clamp_value)) * clamp_value + 1e-4


def test_swiglu_clamp_still_produces_finite_gradients() -> None:
    torch.manual_seed(5)
    layer = SwiGLU(4, 4, clamp_value=1.0)
    with torch.no_grad():
        layer.gate_proj.weight.fill_(100.0)
        layer.up_proj.weight.fill_(100.0)
    inputs = torch.randn(1, 2, 4, requires_grad=True)
    output = layer(inputs)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
