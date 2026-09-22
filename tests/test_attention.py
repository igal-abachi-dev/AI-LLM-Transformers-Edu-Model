import torch
from torch.nn import functional as F

from minifrontier.attention import CausalSelfAttention, manual_scaled_dot_product_attention
from minifrontier.config import ModelConfig
from minifrontier.masking import build_attention_mask
from minifrontier.rope import RoPE


def test_manual_attention_matches_sdpa_output_and_gradients() -> None:
    torch.manual_seed(3)
    query = torch.randn(2, 4, 5, 8, requires_grad=True)
    key = torch.randn(2, 4, 5, 8, requires_grad=True)
    value = torch.randn(2, 4, 5, 8, requires_grad=True)
    references = [tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value)]
    mask = build_attention_mask(5, 5)

    actual = manual_scaled_dot_product_attention(query, key, value, mask=mask)
    expected = F.scaled_dot_product_attention(
        *references,
        attn_mask=mask.unsqueeze(0).unsqueeze(0),
        is_causal=False,
    )
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.sum().backward()
    expected.sum().backward()
    for tensor, reference in zip((query, key, value), references, strict=True):
        assert torch.allclose(tensor.grad, reference.grad, atol=2e-6)


def test_manual_attention_return_weights_is_off_by_default() -> None:
    """Every existing caller (CausalSelfAttention's own manual path included)
    calls this without return_weights -- confirm the return type stays a bare
    tensor, not a tuple, so nothing downstream needs to change."""

    query = torch.randn(1, 1, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    mask = build_attention_mask(3, 3)

    result = manual_scaled_dot_product_attention(query, key, value, mask=mask)
    assert isinstance(result, torch.Tensor)


def test_manual_attention_return_weights_gives_a_real_probability_grid() -> None:
    """The returned weights must be the actual post-softmax attention grid: one
    row per query, summing to 1 over visible keys, and exactly zero on every
    key the causal mask forbids -- not just some arbitrary same-shaped tensor."""

    torch.manual_seed(7)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    mask = build_attention_mask(4, 4)

    output, weights = manual_scaled_dot_product_attention(
        query, key, value, mask=mask, return_weights=True
    )
    assert weights.shape == (1, 2, 4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 2, 4), atol=1e-6)
    forbidden = ~mask.unsqueeze(0).unsqueeze(0).expand_as(weights)
    assert torch.all(weights[forbidden] == 0.0)
    # And the output this call returns must still match the return_weights=False path.
    plain_output = manual_scaled_dot_product_attention(query, key, value, mask=mask)
    assert torch.allclose(output, plain_output)


def test_attention_module_manual_and_sdpa_match() -> None:
    torch.manual_seed(4)
    config = ModelConfig.tiny_edu()
    attention = CausalSelfAttention(config, layer_index=0).eval()
    inputs = torch.randn(2, 6, config.d_model)
    mask = build_attention_mask(6, 6)
    rope = RoPE(config.head_dim, config.max_seq_len)
    cosine, sine = rope(torch.arange(6), dtype=inputs.dtype, device=inputs.device)
    manual = attention(
        inputs,
        cosine,
        sine,
        implementation="manual",
        attention_mask=mask,
    )
    sdpa = attention(inputs, cosine, sine, implementation="sdpa")
    assert torch.allclose(manual, sdpa, atol=2e-6)


def test_attention_is_causal_under_future_perturbation() -> None:
    torch.manual_seed(5)
    config = ModelConfig.tiny_edu()
    attention = CausalSelfAttention(config, layer_index=0).eval()
    original = torch.randn(1, 6, config.d_model)
    changed = original.clone()
    changed[:, -1] += 100
    rope = RoPE(config.head_dim, config.max_seq_len)
    cosine, sine = rope(torch.arange(6), dtype=original.dtype, device=original.device)
    output_a = attention(original, cosine, sine)
    output_b = attention(changed, cosine, sine)
    assert torch.allclose(output_a[:, :-1], output_b[:, :-1], atol=1e-5)


def test_sdpa_full_causal_path_has_no_explicit_mask(monkeypatch) -> None:
    torch.manual_seed(10)
    config = ModelConfig.tiny_edu()
    attention = CausalSelfAttention(config, layer_index=0).eval()
    inputs = torch.randn(1, 4, config.d_model)
    rope = RoPE(config.head_dim, config.max_seq_len)
    cosine, sine = rope(torch.arange(4), dtype=inputs.dtype, device=inputs.device)
    original = F.scaled_dot_product_attention
    observed: dict[str, object] = {}

    def recording_sdpa(query, key, value, **kwargs):
        observed.update(kwargs)
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    attention(inputs, cosine, sine, implementation="sdpa")
    assert observed["attn_mask"] is None
    assert observed["is_causal"] is True
