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
