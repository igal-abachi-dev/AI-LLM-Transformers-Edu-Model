from __future__ import annotations

import sys
from pathlib import Path

import scripts.train_tokenizer as train_tokenizer
from minifrontier.tokenizer import MiniFrontierTokenizer


def _pretokenize(tokenizer: MiniFrontierTokenizer, text: str) -> list[str]:
    return [piece for piece, _ in tokenizer.backend.pre_tokenizer.pre_tokenize_str(text)]


def _run(monkeypatch, tmp_path: Path, corpus_path: Path, output: Path, *extra_args: str) -> None:
    argv = [
        "train_tokenizer.py",
        str(corpus_path),
        "--output",
        str(output),
        "--vocab-size",
        "320",
        "--min-frequency",
        "1",
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_tokenizer.main()


def _write_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(
        "digits 123456789 and more digits, the year 2026 was great " * 20, encoding="utf-8"
    )
    return path


# gpt4's regex caps digit runs at 3 ("123456789" -> "123","456","789"); gpt2's
# does not (ByteLevel's own regex has no digit-length cap) -- the same real
# differentiator `test_pretokenizer_gpt4_caps_digit_runs_at_three` uses in
# tests/test_tokenizer.py, reused here since it's a real, checkable difference
# (unlike contraction-splitting, which both regexes handle the same way).
_GPT4_DIGIT_SPLIT = ["123", "456", "789"]


def test_preset_modern_defaults_to_gpt4_pretokenizer(tmp_path, monkeypatch) -> None:
    corpus = _write_corpus(tmp_path)
    output = tmp_path / "tok"
    _run(monkeypatch, tmp_path, corpus, output, "--preset", "modern")
    tokenizer = MiniFrontierTokenizer.from_directory(output)
    assert _pretokenize(tokenizer, "123456789") == _GPT4_DIGIT_SPLIT


def test_preset_edu_defaults_to_gpt2_pretokenizer(tmp_path, monkeypatch) -> None:
    corpus = _write_corpus(tmp_path)
    output = tmp_path / "tok"
    _run(monkeypatch, tmp_path, corpus, output, "--preset", "edu")
    tokenizer = MiniFrontierTokenizer.from_directory(output)
    assert _pretokenize(tokenizer, "123456789") != _GPT4_DIGIT_SPLIT


def test_explicit_pretokenizer_overrides_preset(tmp_path, monkeypatch) -> None:
    corpus = _write_corpus(tmp_path)
    output = tmp_path / "tok"
    _run(monkeypatch, tmp_path, corpus, output, "--preset", "modern", "--pretokenizer", "gpt2")
    tokenizer = MiniFrontierTokenizer.from_directory(output)
    assert _pretokenize(tokenizer, "123456789") != _GPT4_DIGIT_SPLIT


def test_no_preset_or_pretokenizer_keeps_the_module_default(tmp_path, monkeypatch) -> None:
    corpus = _write_corpus(tmp_path)
    output = tmp_path / "tok"
    _run(monkeypatch, tmp_path, corpus, output)
    tokenizer = MiniFrontierTokenizer.from_directory(output)
    assert _pretokenize(tokenizer, "123456789") != _GPT4_DIGIT_SPLIT
