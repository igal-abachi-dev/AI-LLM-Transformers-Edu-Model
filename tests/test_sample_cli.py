from __future__ import annotations

from pathlib import Path

import scripts.sample as sample
from minifrontier.checkpoint import export_release
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads


def _base_argv(release: Path) -> list[str]:
    return ["sample.py", "--model", str(release), "--prompt", "hi", "--max-new-tokens", "3"]


def _release_with_mtp_heads(tmp_path: Path, mini_tokenizer) -> Path:
    config = ModelConfig.tiny_modern(vocab_size=max(512, mini_tokenizer.vocab_size))
    mtp_heads = MTPHeads(d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=1)
    release = tmp_path / "release"
    export_release(release, MiniFrontier(config), mini_tokenizer, mtp_heads=mtp_heads)
    return release


def test_sample_passes_mtp_heads_by_default_when_release_has_them(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    release = _release_with_mtp_heads(tmp_path, mini_tokenizer)
    captured = {}

    def fake_complete_text(model, tokenizer, prompt, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(sample, "complete_text", fake_complete_text)
    monkeypatch.setattr("sys.argv", _base_argv(release))
    sample.main()
    assert captured["mtp_heads"] is not None


def test_sample_no_speculative_flag_disables_mtp_heads(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    release = _release_with_mtp_heads(tmp_path, mini_tokenizer)
    captured = {}

    def fake_complete_text(model, tokenizer, prompt, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(sample, "complete_text", fake_complete_text)
    monkeypatch.setattr("sys.argv", [*_base_argv(release), "--no-speculative"])
    sample.main()
    assert captured["mtp_heads"] is None


def test_sample_passes_none_when_release_has_no_mtp_heads(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    release = tmp_path / "release"
    export_release(release, MiniFrontier(config), mini_tokenizer)
    captured = {}

    def fake_complete_text(model, tokenizer, prompt, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(sample, "complete_text", fake_complete_text)
    monkeypatch.setattr("sys.argv", _base_argv(release))
    sample.main()
    assert captured["mtp_heads"] is None
