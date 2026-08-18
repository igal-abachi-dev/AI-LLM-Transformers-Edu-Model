import pytest
import torch

from minifrontier.masking import build_attention_mask


def test_full_causal_mask_blocks_future() -> None:
    assert torch.equal(
        build_attention_mask(4, 4),
        torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ),
    )


def test_local_mask_keeps_exact_window() -> None:
    mask = build_attention_mask(5, 5, window_size=2)
    assert mask.sum(dim=-1).tolist() == [1, 2, 2, 2, 2]
    assert mask[-1].tolist() == [False, False, False, True, True]


def test_offset_mask_for_decode_query() -> None:
    mask = build_attention_mask(1, 5, query_start=4, window_size=3)
    assert mask.tolist() == [[False, False, True, True, True]]


def test_mask_rejects_query_beyond_keys() -> None:
    with pytest.raises(ValueError, match="beyond"):
        build_attention_mask(2, 4, query_start=3)


def test_mask_supports_ring_cache_key_offsets() -> None:
    mask = build_attention_mask(
        2,
        6,
        query_start=7,
        key_start=3,
        window_size=4,
    )
    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, True, True, True, True, False],
                [False, False, True, True, True, True],
            ]
        ),
    )
