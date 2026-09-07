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


def _pretokenize(tokenizer: MiniFrontierTokenizer, text: str) -> list[str]:
    return [piece for piece, _ in tokenizer.backend.pre_tokenizer.pre_tokenize_str(text)]


def test_digit_split_none_never_isolates_digit_runs() -> None:
    corpus = ["the year 2026 was great " * 5, "digits 123456789 and more " * 5]
    tokenizer = train_byte_bpe(corpus, vocab_size=300, min_frequency=1, digit_split="none")
    assert _pretokenize(tokenizer, " 2026") == ["Ġ2026"]


def test_train_byte_bpe_default_never_isolates_digit_runs() -> None:
    # Frozen default is "none" (reverted 2026-09-08, see tokenizer.py's docstring):
    # digit-splitting measured only a fertility cost and was never combined with
    # the 16k vocabulary, so callers who don't pass digit_split explicitly must
    # get the same behavior as the real production tokenizer.
    corpus = ["the year 2026 was great " * 5, "digits 123456789 and more " * 5]
    tokenizer = train_byte_bpe(corpus, vocab_size=300, min_frequency=1)
    assert _pretokenize(tokenizer, " 2026") == ["Ġ2026"]


def test_digit_split_no_leading_space_wastes_a_lone_space_token() -> None:
    corpus = ["the year 2026 was great " * 5, "digits 123456789 and more " * 5]
    tokenizer = train_byte_bpe(
        corpus, vocab_size=300, min_frequency=1, digit_split="no_leading_space"
    )
    # Digits split into groups of <=3, but the leading space is its own
    # separate piece -- the exact fertility cost MF-090 measured against the
    # leading_space variant below.
    assert _pretokenize(tokenizer, " 2026") == ["Ġ", "202", "6"]


def test_digit_split_leading_space_keeps_the_space_attached_to_the_digit_group() -> None:
    corpus = ["the year 2026 was great " * 5, "digits 123456789 and more " * 5]
    tokenizer = train_byte_bpe(corpus, vocab_size=300, min_frequency=1, digit_split="leading_space")
    assert _pretokenize(tokenizer, " 2026") == ["Ġ202", "6"]


def test_train_byte_bpe_rejects_unknown_digit_split_mode() -> None:
    with pytest.raises(ValueError, match="digit_split"):
        train_byte_bpe(["abc"], vocab_size=280, min_frequency=1, digit_split="bogus")
