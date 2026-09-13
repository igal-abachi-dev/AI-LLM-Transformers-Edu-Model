"""Behavioral tests for scripts/compare_mtp.py's real-time progress/checkpoint
logging (MF-129) -- not just --help smoke coverage.

MF-129's own motivation: a real retrain using this script gave zero signal for
190+ minutes (no periodic print, no partial checkpoint), indistinguishable
from a hang, and had to be killed unresolved. These tests prove the fix
actually fires during a real (tiny, CPU) training run, not just that the new
CLI flags parse.
"""

from __future__ import annotations

from pathlib import Path

import scripts.compare_mtp as compare_mtp
from minifrontier.data import Document
from minifrontier.shards import TokenShardWriter


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
    for index in range(8):
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


def test_progress_and_partial_checkpoint_fire_during_a_real_run(
    monkeypatch, tmp_path, capsys, mini_tokenizer
):
    shards_path = _build_shards(tmp_path, mini_tokenizer)
    config_path = _write_tiny_config(tmp_path, vocab_size=mini_tokenizer.vocab_size)
    output_path = tmp_path / "output"

    partial_paths_seen: list[Path] = []
    real_save = compare_mtp.save_training_checkpoint

    def spying_save(path, *args, **kwargs):
        partial_paths_seen.append(Path(path))
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr(compare_mtp, "save_training_checkpoint", spying_save)
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_mtp.py",
            "--config",
            str(config_path),
            "--train-shards",
            str(shards_path),
            "--output",
            str(output_path),
            "--updates",
            "10",
            "--batch-size",
            "1",
            "--seed",
            "1",
            "--device",
            "cpu",
            "--arms",
            "mtp",
            "--mtp-extra-heads",
            "1",
            "--mtp-loss-weight",
            "0.3",
            "--progress-interval",
            "3",
            "--checkpoint-interval",
            "4",
        ],
    )

    compare_mtp.main()

    # Real progress lines printed mid-run, not just a final summary.
    captured = capsys.readouterr().out
    assert "3/10 updates" in captured
    assert "6/10 updates" in captured
    assert "9/10 updates" in captured

    # save_training_checkpoint was called for real partial saves DURING
    # training (not only the one final save at the very end) -- this is the
    # actual resilience MF-129 exists for.
    partial_saves = [path for path in partial_paths_seen if path.name.endswith("-partial")]
    assert len(partial_saves) >= 2, partial_paths_seen

    # The final, real checkpoint exists...
    assert (output_path / "seed-1-mtp" / "model.safetensors").exists()
    # ...and the partial one is cleaned up once the arm finishes successfully,
    # rather than left behind as stale, superseded state.
    assert not (output_path / "seed-1-mtp-partial").exists()


def test_progress_and_checkpoint_interval_are_validated(monkeypatch) -> None:
    import pytest

    # --config/--train-shards/--output point at nothing real: the validation
    # this test checks fires before any file is ever touched, so that is fine.
    base_argv = [
        "compare_mtp.py",
        "--config",
        "unused.toml",
        "--train-shards",
        "unused",
        "--output",
        "unused-out",
    ]
    monkeypatch.setattr("sys.argv", [*base_argv, "--progress-interval", "0"])
    with pytest.raises(ValueError, match="progress-interval must be positive"):
        compare_mtp.main()

    monkeypatch.setattr("sys.argv", [*base_argv, "--checkpoint-interval", "0"])
    with pytest.raises(ValueError, match="checkpoint-interval must be positive"):
        compare_mtp.main()
