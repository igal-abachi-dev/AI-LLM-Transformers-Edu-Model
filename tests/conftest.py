from pathlib import Path

import pytest

from minifrontier.tokenizer import MiniFrontierTokenizer, train_byte_bpe


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tokenizer")
    corpus = [
        "MiniFrontier teaches attention from first principles.\n",
        "def add(left, right):\n    return left + right\n",
        "שלום עולם — Καλημέρα κόσμε — こんにちは世界 — hello world 🌍\n",
        "The quick brown fox jumps over the lazy dog.\n" * 8,
    ]
    tokenizer = train_byte_bpe(corpus, vocab_size=320, min_frequency=1)
    tokenizer.save(directory)
    return directory


@pytest.fixture(scope="session")
def mini_tokenizer(tokenizer_dir: Path) -> MiniFrontierTokenizer:
    return MiniFrontierTokenizer.from_directory(tokenizer_dir)
