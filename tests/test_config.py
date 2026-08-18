from dataclasses import replace
from pathlib import Path

import pytest

from minifrontier.config import ModelConfig

ROOT = Path(__file__).parents[1]


def expected_edu_parameters(config: ModelConfig) -> int:
    embedding = config.vocab_size * config.d_model
    attention = 4 * config.d_model * config.d_model
    feed_forward = 3 * config.d_model * config.d_ff
    block_norms = 2 * config.d_model
    return embedding + config.n_layers * (attention + feed_forward + block_norms) + config.d_model


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("50m-edu.toml", 53_361_152), ("150m-edu.toml", 154_172_160)],
)
def test_frozen_edu_parameter_targets(filename: str, expected: int) -> None:
    config = ModelConfig.from_toml(ROOT / "configs" / filename)
    assert expected_edu_parameters(config) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "50m-edu.toml",
        "50m-modern.toml",
        "150m-edu.toml",
        "150m-modern.toml",
        "350m-modern.toml",
        "500m-modern.toml",
    ],
)
def test_all_frozen_presets_validate(filename: str) -> None:
    config = ModelConfig.from_toml(ROOT / "configs" / filename)
    assert config.head_dim == 64


def test_config_rejects_incompatible_heads() -> None:
    with pytest.raises(ValueError, match="d_model must be divisible"):
        ModelConfig.tiny_edu(d_model=30, n_heads=4)


def test_config_rejects_edu_gqa() -> None:
    with pytest.raises(ValueError, match="Edu requires MHA"):
        ModelConfig(n_kv_heads=4)


def test_config_rejects_unknown_attention_implementation() -> None:
    with pytest.raises(ValueError, match="attention_impl"):
        ModelConfig(attention_impl="fastest")  # type: ignore[arg-type]


def test_hybrid_flex_dropout_contract_fails_at_configuration_time() -> None:
    config = ModelConfig.tiny_modern(attention_impl="auto")
    with pytest.raises(ValueError, match="FlexAttention requires dropout=0"):
        replace(config, dropout=0.1)


def test_hybrid_schedule_is_three_local_then_global() -> None:
    config = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        n_layers=8,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=96,
        qk_norm=True,
        attention_pattern="hybrid",
        local_window=8,
        preset="modern",
    )
    assert [config.is_local_layer(index) for index in range(8)] == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    ]
