from __future__ import annotations

from pathlib import Path

import pytest
import torch

import scripts.export as export
from minifrontier.checkpoint import load_release, load_release_mtp_heads, save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.ema import EMAWeights
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.release import verify_release


def _write_checkpoint(directory: Path, config: ModelConfig, *, mtp_extra_heads: int) -> None:
    model = MiniFrontier(config)
    mtp_heads = None
    if mtp_extra_heads > 0:
        mtp_heads = MTPHeads(
            d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=mtp_extra_heads
        )
    save_training_checkpoint(
        directory,
        model,
        trainer_state={
            "training_config": {"mtp_extra_heads": mtp_extra_heads, "mtp_loss_weight": 0.3},
        },
        mtp_heads=mtp_heads,
    )


def test_export_forwards_mtp_heads_when_checkpoint_was_trained_with_them(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    config = ModelConfig.tiny_modern(vocab_size=max(512, mini_tokenizer.vocab_size))
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, config, mtp_extra_heads=1)
    tokenizer_dir = tmp_path / "tokenizer"
    mini_tokenizer.save(tokenizer_dir)
    output = tmp_path / "release"

    monkeypatch.setattr(
        "sys.argv",
        [
            "export.py",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--output",
            str(output),
            "--keep-source",
        ],
    )
    export.main()

    report = verify_release(output)
    assert report["mtp_heads_present"] is True
    assert load_release_mtp_heads(output, config) is not None
    assert checkpoint.exists()  # --keep-source honored


def test_export_omits_mtp_heads_when_checkpoint_has_none(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, config, mtp_extra_heads=0)
    tokenizer_dir = tmp_path / "tokenizer"
    mini_tokenizer.save(tokenizer_dir)
    output = tmp_path / "release"

    monkeypatch.setattr(
        "sys.argv",
        [
            "export.py",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--output",
            str(output),
            "--keep-source",
        ],
    )
    export.main()

    report = verify_release(output)
    assert report["mtp_heads_present"] is False
    assert not (output / "mtp_heads.safetensors").exists()


def test_export_handles_checkpoint_with_no_mtp_extra_heads_key_at_all(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    """Older checkpoints saved before MTP existed have no 'mtp_extra_heads' key
    in trainer_state.json at all -- export must default to 0/None, not crash."""

    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, MiniFrontier(config))  # no trainer_state at all
    tokenizer_dir = tmp_path / "tokenizer"
    mini_tokenizer.save(tokenizer_dir)
    output = tmp_path / "release"

    monkeypatch.setattr(
        "sys.argv",
        [
            "export.py",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--output",
            str(output),
            "--keep-source",
        ],
    )
    export.main()

    report = verify_release(output)
    assert report["mtp_heads_present"] is False


def test_export_weights_ema_publishes_the_real_shadow_not_the_live_weights(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    """MF-086 part 4 follow-up: `--weights ema` must publish the EMA shadow,
    not the live weights. The shadow is deliberately shifted to a real,
    distinguishing value (not left at its freshly-initialized-equal-to-live
    state) so the two are provably different, not coincidentally the same."""

    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    model = MiniFrontier(config)
    ema = EMAWeights(model, decay=0.999)
    with torch.no_grad():
        for shadow in ema.state_dict().values():
            shadow.add_(1.0)
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, model, ema=ema)
    tokenizer_dir = tmp_path / "tokenizer"
    mini_tokenizer.save(tokenizer_dir)
    output = tmp_path / "release"

    monkeypatch.setattr(
        "sys.argv",
        [
            "export.py",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--output",
            str(output),
            "--weights",
            "ema",
            "--keep-source",
        ],
    )
    export.main()

    released_model, _ = load_release(output)
    live_state = model.state_dict()
    for name, released_value in released_model.state_dict().items():
        assert torch.equal(released_value, live_state[name] + 1.0), name


def test_export_weights_ema_rejects_a_checkpoint_saved_without_ema(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, MiniFrontier(config))  # no ema=
    tokenizer_dir = tmp_path / "tokenizer"
    mini_tokenizer.save(tokenizer_dir)
    output = tmp_path / "release"

    monkeypatch.setattr(
        "sys.argv",
        [
            "export.py",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--output",
            str(output),
            "--weights",
            "ema",
        ],
    )
    with pytest.raises(ValueError, match="not saved with EMA tracking"):
        export.main()
