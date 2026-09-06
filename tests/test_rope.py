import math

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from minifrontier.rope import RoPE, apply_rotary, rotate_half


def test_rotate_half_uses_split_half_convention() -> None:
    inputs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(rotate_half(inputs), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_rope_position_zero_is_identity_and_preserves_norm() -> None:
    rope = RoPE(head_dim=8, max_seq_len=16)
    positions = torch.arange(4)
    cosine, sine = rope(positions, dtype=torch.float32, device=torch.device("cpu"))
    inputs = torch.randn(2, 3, 4, 8)
    output = apply_rotary(inputs, cosine, sine)
    assert torch.equal(output[:, :, 0], inputs[:, :, 0])
    assert torch.allclose(output.norm(dim=-1), inputs.norm(dim=-1), atol=1e-6)
    assert output.dtype == inputs.dtype


def test_rope_known_first_frequency_rotation() -> None:
    rope = RoPE(head_dim=4, max_seq_len=4)
    cosine, sine = rope(torch.tensor([1]), dtype=torch.float64, device=torch.device("cpu"))
    inputs = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float64)
    output = apply_rotary(inputs, cosine, sine)
    assert math.isclose(output[0, 0, 0, 0].item(), math.cos(1.0), abs_tol=1e-7)
    assert math.isclose(output[0, 0, 0, 2].item(), math.sin(1.0), abs_tol=1e-7)


def test_rope_offset_positions_are_stable() -> None:
    rope = RoPE(head_dim=8, max_seq_len=16)
    all_cos, all_sin = rope(torch.arange(8), dtype=torch.float32, device=torch.device("cpu"))
    sub_cos, sub_sin = rope(torch.arange(3, 8), dtype=torch.float32, device=torch.device("cpu"))
    assert torch.equal(all_cos[3:], sub_cos)
    assert torch.equal(all_sin[3:], sub_sin)


def test_apply_rotary_computes_in_fp32_before_casting_back_to_input_dtype() -> None:
    """BF16/FP16 cos/sin are only ~0.4% precise in [-1, 1]; the multiply must not
    happen at that precision, or every Q/K in every layer inherits the error."""

    torch.manual_seed(4)
    head_dim = 8
    inputs = torch.randn(2, 3, 5, head_dim, dtype=torch.bfloat16)
    cosine = torch.rand(5, head_dim, dtype=torch.float32)
    sine = torch.rand(5, head_dim, dtype=torch.float32)

    result = apply_rotary(inputs, cosine, sine)

    expected = (
        inputs.float() * cosine.unsqueeze(0).unsqueeze(0)
        + rotate_half(inputs.float()) * sine.unsqueeze(0).unsqueeze(0)
    ).to(torch.bfloat16)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, expected)

    # The naive bf16-native multiply (rounding cos/sin down first) is a
    # different, less accurate answer -- this pins down that we take the more
    # accurate path, not merely "a" path.
    naive = inputs * cosine.to(torch.bfloat16).unsqueeze(0).unsqueeze(0) + rotate_half(
        inputs
    ) * sine.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
    assert not torch.equal(result, naive)


def test_apply_rotary_never_downcasts_an_already_high_precision_caller() -> None:
    rope = RoPE(head_dim=4, max_seq_len=4)
    cosine, sine = rope(torch.tensor([1]), dtype=torch.float64, device=torch.device("cpu"))
    inputs = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float64)
    output = apply_rotary(inputs, cosine, sine)
    assert output.dtype == torch.float64


def test_rope_matches_transformers_llama_primitive() -> None:
    """External ground truth catches a self-consistent split/adjacent-pair mistake."""

    torch.manual_seed(9)
    rope = RoPE(head_dim=8, max_seq_len=16)
    cosine, sine = rope(torch.arange(6), dtype=torch.float32, device=torch.device("cpu"))
    query = torch.randn(2, 3, 6, 8)
    key = torch.randn(2, 3, 6, 8)
    actual_query = apply_rotary(query, cosine, sine)
    actual_key = apply_rotary(key, cosine, sine)
    expected_query, expected_key = apply_rotary_pos_emb(
        query,
        key,
        cosine,
        sine,
        unsqueeze_dim=0,
    )
    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)
