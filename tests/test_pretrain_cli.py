"""Behavioral tests for train/pretrain.py's run() -- not just --help smoke coverage.

train/pretrain.py has no package __init__.py, so it is loaded directly from its file path
rather than imported by dotted name; this mirrors how the script itself is actually invoked.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

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
        mixture=None,
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
        keep_last_n_checkpoints=None,
        loss_chunk_size=None,
        z_loss_weight=0.0,
        mtp_extra_heads=0,
        mtp_loss_weight=0.0,
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


def test_loss_chunk_size_and_z_loss_weight_flags_reach_real_training(
    tmp_path, mini_tokenizer
) -> None:
    """Regression for a real gap: MF-084's chunked loss/z-loss existed in
    TrainingConfig but had no CLI path from the actual training entry point,
    so a real run could never reach them regardless of vocabulary size."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "chunked"
    state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            no_checkpoint=True,
            loss_chunk_size=2,
            z_loss_weight=1e-4,
        )
    )
    assert state.completed_updates == 2
    assert state.last_loss is not None and state.last_loss == state.last_loss  # not NaN


def test_resume_with_mtp_extra_heads_is_rejected(tmp_path, mini_tokenizer) -> None:
    """MTP head weights are not part of the saved checkpoint (see mtp.py) --
    combining --resume with MTP must fail loudly, not silently reinitialize."""

    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    try:
        pretrain.run(
            _args(
                config_path,
                shards_path,
                tmp_path / "out",
                resume=tmp_path,
                mtp_extra_heads=1,
                mtp_loss_weight=0.5,
            )
        )
    except ValueError as error:
        assert "resume" in str(error) and "mtp" in str(error).lower()
    else:
        raise AssertionError("expected a ValueError for --resume with mtp_extra_heads > 0")


def test_mtp_extra_heads_flag_reaches_real_training(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "mtp"
    state, _ = pretrain.run(
        _args(
            config_path,
            shards_path,
            output,
            no_checkpoint=True,
            mtp_extra_heads=1,
            mtp_loss_weight=0.5,
        )
    )
    assert state.completed_updates == 2
    assert state.last_loss is not None and state.last_loss == state.last_loss  # not NaN


def test_mixture_flag_reaches_real_training_and_checkpoint_reloads(
    tmp_path, mini_tokenizer
) -> None:
    """Real end-to-end wiring: --mixture -> real training -> a real checkpoint
    whose mixture cursor state round-trips. MixtureBatchProvider's own exact-
    resume-continues-not-restarts guarantee is unit-tested directly against the
    class in test_shards.py; pretrain.py's CLI has no way to run a real partial-
    then-resume-to-completion scenario (each invocation always runs to its full
    `--updates` in one call), so this test verifies the wiring, not the resume
    algorithm itself."""

    web_shards = _build_shards(tmp_path / "web-src", mini_tokenizer)
    code_shards = _build_shards(tmp_path / "code-src", mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    output = tmp_path / "mixture-run"
    mixture = [f"web;{web_shards};0.7", f"code;{code_shards};0.3"]

    state, _ = pretrain.run(
        _args(
            config_path,
            None,
            output,
            train_shards=None,
            mixture=mixture,
            no_checkpoint=False,
        )
    )
    assert state.completed_updates == 2
    assert state.last_loss is not None and state.last_loss == state.last_loss  # not NaN

    # Loading the real checkpoint back (same config, same mixture) must succeed --
    # this is what --resume actually does, exercised for real rather than assumed.
    reloaded_state, _ = pretrain.run(
        _args(
            config_path,
            None,
            tmp_path / "mixture-reload",
            train_shards=None,
            mixture=mixture,
            resume=output / "final",
            no_checkpoint=True,
        )
    )
    assert reloaded_state.completed_updates == 2  # already at max_updates=2, correctly a no-op


def test_mixture_and_train_shards_are_mutually_exclusive(tmp_path, mini_tokenizer) -> None:
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, mini_tokenizer.vocab_size)
    with pytest.raises(ValueError, match="exactly one"):
        pretrain.run(
            _args(
                config_path,
                shards_path,
                tmp_path / "out",
                mixture=[f"web;{shards_path};1.0"],
                no_checkpoint=True,
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        pretrain.run(
            _args(config_path, None, tmp_path / "out", train_shards=None, no_checkpoint=True)
        )
