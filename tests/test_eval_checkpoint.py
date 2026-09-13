from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from minifrontier.checkpoint import save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.ema import EMAWeights
from minifrontier.model import MiniFrontier
from minifrontier.shards import PackedShardDataset, TokenShardWriter
from scripts.eval_checkpoint import evaluate_checkpoint
from tests.test_shards import make_document


def _write_validation_shards(directory: Path, tokenizer) -> None:
    writer = TokenShardWriter(directory, tokenizer, sequence_length=8, sequences_per_shard=2)
    for index in range(4):
        writer.add(make_document(f"validation document number {index} has real words", str(index)))
    writer.finalize(drop_remainder=False)
    assert len(PackedShardDataset(directory)) > 0


def test_evaluate_checkpoint_reports_real_finite_metrics(
    tmp_path, mini_tokenizer, tokenizer_dir
) -> None:
    config = ModelConfig.tiny_modern(vocab_size=mini_tokenizer.vocab_size, max_seq_len=8)
    checkpoint_dir = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint_dir, MiniFrontier(config))

    validation_dir = tmp_path / "validation"
    _write_validation_shards(validation_dir, mini_tokenizer)

    result = evaluate_checkpoint(
        checkpoint_dir, validation_dir, tokenizer_dir, batch_size=2, device="cpu"
    )

    assert result["checkpoint"]
    assert math.isfinite(result["cross_entropy"])
    assert math.isfinite(result["perplexity"])
    assert math.isfinite(result["bits_per_byte"])
    assert result["predicted_tokens"] > 0
    assert result["utf8_bytes"] > 0


def test_evaluate_checkpoint_rejects_config_that_no_longer_matches_saved_weights(
    tmp_path, mini_tokenizer, tokenizer_dir
) -> None:
    config = ModelConfig.tiny_modern(vocab_size=mini_tokenizer.vocab_size, max_seq_len=8)
    checkpoint_dir = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint_dir, MiniFrontier(config))

    validation_dir = tmp_path / "validation"
    _write_validation_shards(validation_dir, mini_tokenizer)

    # A corrupted/edited config.json that no longer matches the shapes actually
    # saved in model.safetensors must fail loudly, not silently load garbage.
    other_config = ModelConfig.tiny_modern(
        vocab_size=mini_tokenizer.vocab_size, max_seq_len=8, d_model=64
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(other_config.to_dict()), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        evaluate_checkpoint(checkpoint_dir, validation_dir, tokenizer_dir, device="cpu")


def test_weights_ema_evaluates_the_real_shadow_not_the_live_weights(
    tmp_path, mini_tokenizer, tokenizer_dir
) -> None:
    """MF-086 part 4 follow-up: `--weights ema` must actually swap in the EMA
    shadow before evaluating, not silently evaluate the live weights while
    claiming otherwise. Made the shadow's own weights real and deliberately
    different from the live ones (not just freshly initialized, which could
    coincidentally match) so the two evaluations are provably distinguishable."""

    config = ModelConfig.tiny_modern(vocab_size=mini_tokenizer.vocab_size, max_seq_len=8)
    model = MiniFrontier(config)
    ema = EMAWeights(model, decay=0.999)
    with torch.no_grad():
        for shadow in ema.state_dict().values():
            shadow.add_(1.0)  # real, large, deliberately-distinguishing shift
    checkpoint_dir = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint_dir, model, ema=ema)
    assert (checkpoint_dir / "ema.safetensors").exists()

    validation_dir = tmp_path / "validation"
    _write_validation_shards(validation_dir, mini_tokenizer)

    live_result = evaluate_checkpoint(
        checkpoint_dir, validation_dir, tokenizer_dir, batch_size=2, device="cpu", weights="live"
    )
    ema_result = evaluate_checkpoint(
        checkpoint_dir, validation_dir, tokenizer_dir, batch_size=2, device="cpu", weights="ema"
    )
    assert live_result["weights"] == "live"
    assert ema_result["weights"] == "ema"
    # Real, different models -> real, different (finite) cross-entropy.
    assert math.isfinite(ema_result["cross_entropy"])
    assert ema_result["cross_entropy"] != live_result["cross_entropy"]


def test_weights_ema_rejects_a_checkpoint_saved_without_ema(
    tmp_path, mini_tokenizer, tokenizer_dir
) -> None:
    config = ModelConfig.tiny_modern(vocab_size=mini_tokenizer.vocab_size, max_seq_len=8)
    checkpoint_dir = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint_dir, MiniFrontier(config))  # no ema=

    validation_dir = tmp_path / "validation"
    _write_validation_shards(validation_dir, mini_tokenizer)

    with pytest.raises(ValueError, match="not saved with EMA tracking"):
        evaluate_checkpoint(
            checkpoint_dir, validation_dir, tokenizer_dir, device="cpu", weights="ema"
        )
