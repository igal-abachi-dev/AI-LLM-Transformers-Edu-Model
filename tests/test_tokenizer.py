import hashlib

import pytest

from minifrontier.tokenizer import (
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    MiniFrontierTokenizer,
    train_byte_bpe,
)


def test_special_tokens_have_frozen_atomic_ids(mini_tokenizer) -> None:
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        assert mini_tokenizer.backend.token_to_id(token) == expected_id
        assert mini_tokenizer.encode(token) == [expected_id]


@pytest.mark.parametrize(
    "text",
    [
        "plain ASCII",
        "שלום עולם",
        "こんにちは世界 🌍",
        "def f(x):\n\treturn x ** 2\n",
        "embedded\x01control",
    ],
)
def test_arbitrary_unicode_and_code_round_trip(mini_tokenizer, text: str) -> None:
    token_ids = mini_tokenizer.encode(text)
    assert token_ids
    assert mini_tokenizer.decode(token_ids) == text


def test_bos_eos_are_only_added_explicitly(mini_tokenizer) -> None:
    plain = mini_tokenizer.encode("hello")
    bounded = mini_tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert bounded == [mini_tokenizer.bos_id, *plain, mini_tokenizer.eos_id]


def test_training_is_deterministic_for_fixed_order(tmp_path) -> None:
    corpus = ["alpha beta gamma" * 8, "delta epsilon" * 8]
    hashes = []
    for index in range(2):
        directory = tmp_path / str(index)
        train_byte_bpe(corpus, vocab_size=280, min_frequency=1).save(directory)
        hashes.append(hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


def test_save_rejects_invalid_model_max_length(tmp_path, mini_tokenizer) -> None:
    with pytest.raises(ValueError, match="model_max_length"):
        mini_tokenizer.save(tmp_path, model_max_length=0)


def test_bpe_compresses_repetitive_text(mini_tokenizer) -> None:
    text = "attention transformer attention transformer " * 20
    assert len(mini_tokenizer.encode(text)) < len(text.encode("utf-8"))


def test_loader_rejects_changed_tokenizer_hash(tokenizer_dir) -> None:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    original = tokenizer_path.read_bytes()
    try:
        tokenizer_path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="hash"):
            MiniFrontierTokenizer.from_directory(tokenizer_dir)
    finally:
        tokenizer_path.write_bytes(original)


def test_contract_contains_all_expected_reserved_tokens() -> None:
    assert len(SPECIAL_TOKENS) == 11
    assert SPECIAL_TOKENS[0:3] == ("<|pad|>", "<|bos|>", "<|eos|>")
