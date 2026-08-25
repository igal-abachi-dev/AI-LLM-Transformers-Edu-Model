"""Behavioral tests for train/pretrain.py's run() -- not just --help smoke coverage.

train/pretrain.py has no package __init__.py, so it is loaded directly from its file path
rather than imported by dotted name; this mirrors how the script itself is actually invoked.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from minifrontier.data import Document
from minifrontier.shards import TokenShardWriter

_PRETRAIN_PATH = Path(__file__).resolve().parents[1] / "train" / "pretrain.py"
_spec = importlib.util.spec_from_file_location("minifrontier_train_pretrain", _PRETRAIN_PATH)
assert _spec is not None and _spec.loader is not None
pretrain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pretrain)


def _make_document(text: str, record_id: str) -> Document:
    return Document.create(
        text,
        source="fixture",
        revision="abc123",
        license="Apache-2.0",
        language="en",
        record_id=record_id,
    )


def _build_shards(tmp_path: Path, tokenizer) -> Path:
    writer = TokenShardWriter(
        tmp_path / "train", tokenizer, sequence_length=6, sequences_per_shard=4
    )
    for index in range(6):
        writer.add(_make_document(f"tiny document number {index} with some words", str(index)))
    writer.finalize(drop_remainder=False)
    return tmp_path / "train"


def _write_tiny_config(tmp_path: Path, vocab_size: int) -> Path:
    config_path = tmp_path / "tiny.toml"
    config_path.write_text(
        "\n".join(
            [
                'preset = "edu"',
                f"vocab_size = {vocab_size}",
                "max_seq_len = 16",
                "n_layers = 1",
                "d_model = 16",
                "n_heads = 2",
                "n_kv_heads = 2",
                "d_ff = 32",
                "norm_eps = 1e-6",
                "rope_theta = 10000.0",
                "qk_norm = false",
                'attention_pattern = "full"',
                "local_window = 16",
                'global_position_encoding = "rope"',
                "dropout = 0.0",
                "tie_embeddings = true",
                'attention_impl = "sdpa"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _args(
    config_path: Path, shards_path: Path, output_path: Path, **overrides
) -> argparse.Namespace:
    values = dict(
        config=config_path,
        train_shards=shards_path,
        output=output_path,
        resume=None,
        device="cpu",
        seed=42,
        precision="float32",
        attention_impl=None,
        updates=2,
        warmup_updates=0,
        batch_size=1,
        accumulation_steps=1,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=1.0,
        checkpoint_interval=1,
        no_checkpoint=False,
        activation_checkpointing=False,
        compile=False,
        compile_backend=None,
        compile_fail=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_no_checkpoint_flag_skips_all_checkpoint_writes(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)

    # Default behavior: checkpoint-interval=1 with 2 updates writes at least one
    # interval checkpoint plus the final one.
    checkpointed_output = tmp_path / "with-checkpoints"
    pretrain.run(_args(config_path, shards_path, checkpointed_output))
    assert (checkpointed_output / "final" / "model.safetensors").exists()
    assert list(checkpointed_output.glob("checkpoint-*"))
    assert (checkpointed_output / "run.json").exists()

    # --no-checkpoint: only the report is written, no model/optimizer state at all.
    bare_output = tmp_path / "no-checkpoints"
    pretrain.run(_args(config_path, shards_path, bare_output, no_checkpoint=True, seed=43))
    assert (bare_output / "run.json").exists()
    assert not (bare_output / "final").exists()
    assert not list(bare_output.glob("checkpoint-*"))
    assert list(bare_output.iterdir()) == [bare_output / "run.json"]
