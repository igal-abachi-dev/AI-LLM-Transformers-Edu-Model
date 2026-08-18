import pytest

from minifrontier.config import ModelConfig
from minifrontier.overfit import run_overfit


@pytest.mark.slow
def test_tiny_edu_overfits_one_hundred_pattern_examples() -> None:
    config = ModelConfig.tiny_edu(n_layers=1, d_model=24, n_heads=4, d_ff=64)
    _, result, metadata = run_overfit(
        config,
        examples=100,
        sequence_length=8,
        steps=700,
        learning_rate=5e-3,
    )
    assert result.final_loss < 1e-3
    assert metadata.train_tokens == 100 * 7 * 700


@pytest.mark.slow
def test_tiny_modern_overfits_pattern_examples() -> None:
    config = ModelConfig.tiny_modern(
        vocab_size=64,
        max_seq_len=16,
        n_layers=4,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=96,
        local_window=8,
        attention_impl="sdpa",
    )
    _, result, metadata = run_overfit(
        config,
        examples=32,
        sequence_length=8,
        steps=700,
        learning_rate=5e-3,
    )
    assert result.final_loss < 1e-3
    assert metadata.name == "modern-overfit"
