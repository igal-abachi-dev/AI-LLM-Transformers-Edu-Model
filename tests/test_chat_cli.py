from __future__ import annotations

from pathlib import Path

import scripts.chat as chat_cli
from minifrontier.checkpoint import export_release
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def test_turn_seed_varies_by_turn_but_is_reproducible_given_the_same_base_seed() -> None:
    first_turn = chat_cli._turn_seed(42, 0)
    second_turn = chat_cli._turn_seed(42, 1)
    assert first_turn != second_turn
    # Same base seed, same turn index -> the exact same derived seed, every time.
    assert chat_cli._turn_seed(42, 0) == first_turn
    # A different base seed changes the whole derived sequence.
    assert chat_cli._turn_seed(7, 0) != first_turn


def _release(tmp_path: Path, mini_tokenizer) -> Path:
    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    release = tmp_path / "release"
    export_release(release, MiniFrontier(config), mini_tokenizer)
    return release


def test_chat_cli_derives_a_different_seed_each_turn(tmp_path, mini_tokenizer, monkeypatch) -> None:
    """Real, reproduced UX gap: passing the same fixed `--seed` to every turn
    (the prior behavior) meant the same question always got the same reply,
    since `complete_text`'s own torch.Generator starts fresh from that seed on
    every call."""

    release = _release(tmp_path, mini_tokenizer)
    captured_seeds = []

    def fake_generate_assistant(model, tokenizer, messages, **kwargs):
        captured_seeds.append(kwargs["seed"])
        return "ok"

    prompts = iter(["first question", "second question", ""])
    monkeypatch.setattr(chat_cli, "generate_assistant", fake_generate_assistant)
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    monkeypatch.setattr("sys.argv", ["chat.py", "--model", str(release), "--seed", "42"])
    chat_cli.main()

    assert len(captured_seeds) == 2
    assert captured_seeds[0] != captured_seeds[1]
    assert captured_seeds[0] == chat_cli._turn_seed(42, 0)
    assert captured_seeds[1] == chat_cli._turn_seed(42, 1)
