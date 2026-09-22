from __future__ import annotations

from pathlib import Path

import torch

import scripts.inspect_attention as inspect_attention
from minifrontier.attention import manual_scaled_dot_product_attention
from minifrontier.checkpoint import export_release
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def _release(tmp_path: Path, mini_tokenizer) -> Path:
    torch.manual_seed(0)
    config = ModelConfig.tiny_modern(vocab_size=mini_tokenizer.vocab_size, max_seq_len=64)
    release = tmp_path / "release"
    export_release(release, MiniFrontier(config), mini_tokenizer)
    return release


def test_inspect_attention_runs_end_to_end_and_restores_the_real_function(
    tmp_path, mini_tokenizer, monkeypatch, capsys
) -> None:
    release = _release(tmp_path, mini_tokenizer)
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect_attention.py",
            "--model",
            str(release),
            "--prompt",
            "hello world",
            "--top-k",
            "2",
        ],
    )
    inspect_attention.main()
    output = capsys.readouterr().out
    assert "Layer 0" in output
    assert "head 0" in output
    # The monkeypatch inside main() must be undone once it returns -- confirm the
    # module-level name is back to the real function, not the capturing wrapper.
    import minifrontier.attention as attention_module

    assert (
        attention_module.manual_scaled_dot_product_attention is manual_scaled_dot_product_attention
    )


def test_inspect_attention_rejects_out_of_range_position(
    tmp_path, mini_tokenizer, monkeypatch
) -> None:
    release = _release(tmp_path, mini_tokenizer)
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect_attention.py",
            "--model",
            str(release),
            "--prompt",
            "hi",
            "--position",
            "999",
        ],
    )
    try:
        inspect_attention.main()
        raised = False
    except SystemExit:
        raised = True
    assert raised
