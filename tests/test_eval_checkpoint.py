from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from minifrontier.checkpoint import save_training_checkpoint
from minifrontier.config import ModelConfig
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
