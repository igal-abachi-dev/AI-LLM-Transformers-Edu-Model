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


def test_tokenizer_missing_a_newer_special_token_still_loads() -> None:
    """An already-published tokenizer trained before <|eot|>/<|file_sep|>/
    <|repo_name|> existed (MF-103) must remain loadable: growing
    SPECIAL_TOKENS must not make every already-released checkpoint's own
    tokenizer permanently unloadable. The model trained against an old
    tokenizer never learned those tokens either, so their plain absence is
    not drift."""

    import minifrontier.tokenizer as tokenizer_module

    original_tokens = tokenizer_module.SPECIAL_TOKENS
    original_ids = tokenizer_module.SPECIAL_TOKEN_IDS
    try:
        tokenizer_module.SPECIAL_TOKENS = original_tokens[:11]
        tokenizer_module.SPECIAL_TOKEN_IDS = {
            token: index for index, token in enumerate(original_tokens[:11])
        }
        old_style = train_byte_bpe(["hello world " * 10], vocab_size=300, min_frequency=1)
    finally:
        tokenizer_module.SPECIAL_TOKENS = original_tokens
        tokenizer_module.SPECIAL_TOKEN_IDS = original_ids

    # Reload the SAME backend object under the CURRENT (14-token) contract --
    # this is the real check: __init__ re-runs _validate_special_tokens.
    reloaded = MiniFrontierTokenizer(old_style.backend)
    assert reloaded.backend.token_to_id("<|eot|>") is None
    assert reloaded.eos_id == SPECIAL_TOKEN_IDS["<|eos|>"]


def test_tokenizer_still_rejects_a_special_token_present_at_the_wrong_id() -> None:
    """Real drift -- a token that DOES exist but at a different ID than the
    current contract expects -- must still be rejected unconditionally."""

    import minifrontier.tokenizer as tokenizer_module

    original_tokens = tokenizer_module.SPECIAL_TOKENS
    original_ids = tokenizer_module.SPECIAL_TOKEN_IDS
    try:
        # Swap <|bos|> and <|eos|> so <|eos|> lands at ID 1, not 2 -- a real
        # token, present, at the wrong position relative to the real contract.
        swapped = list(original_tokens)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        tokenizer_module.SPECIAL_TOKENS = tuple(swapped)
        tokenizer_module.SPECIAL_TOKEN_IDS = {token: index for index, token in enumerate(swapped)}
        drifted = train_byte_bpe(["hello world " * 10], vocab_size=300, min_frequency=1)
    finally:
        tokenizer_module.SPECIAL_TOKENS = original_tokens
        tokenizer_module.SPECIAL_TOKEN_IDS = original_ids

    with pytest.raises(ValueError, match="must have ID"):
        MiniFrontierTokenizer(drifted.backend)


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


def test_eot_id_is_distinct_from_eos_id(mini_tokenizer) -> None:
    """MF-103: <|eot|> (chat/SFT turn boundary) must never collide with
    <|eos|> (pretraining document boundary) -- that collision is the whole
    problem this task exists to fix."""

    assert mini_tokenizer.eot_id == SPECIAL_TOKEN_IDS["<|eot|>"]
    assert mini_tokenizer.eot_id != mini_tokenizer.eos_id
    assert mini_tokenizer.encode("<|eot|>") == [mini_tokenizer.eot_id]


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
    assert len(SPECIAL_TOKENS) == 14
    assert SPECIAL_TOKENS[0:3] == ("<|pad|>", "<|bos|>", "<|eos|>")
    assert SPECIAL_TOKENS[11:14] == ("<|eot|>", "<|file_sep|>", "<|repo_name|>")


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


def _gpt4_corpus() -> list[str]:
    return [
        "the year 2026 was great " * 5,
        "digits 123456789 and more digits " * 5,
        "don't can't I'm it's " * 5,
    ]


def test_pretokenizer_default_is_gpt2_and_unchanged() -> None:
    tokenizer = train_byte_bpe(_gpt4_corpus(), vocab_size=320, min_frequency=1)
    explicit = train_byte_bpe(_gpt4_corpus(), vocab_size=320, min_frequency=1, pretokenizer="gpt2")
    assert _pretokenize(tokenizer, " 2026") == _pretokenize(explicit, " 2026")


def test_pretokenizer_gpt4_caps_digit_runs_at_three() -> None:
    tokenizer = train_byte_bpe(_gpt4_corpus(), vocab_size=320, min_frequency=1, pretokenizer="gpt4")
    assert _pretokenize(tokenizer, "123456789") == ["123", "456", "789"]


def test_pretokenizer_gpt4_splits_contractions() -> None:
    tokenizer = train_byte_bpe(_gpt4_corpus(), vocab_size=320, min_frequency=1, pretokenizer="gpt4")
    assert _pretokenize(tokenizer, "don't") == ["don", "'t"]


def test_pretokenizer_gpt4_still_round_trips_arbitrary_text() -> None:
    tokenizer = train_byte_bpe(_gpt4_corpus(), vocab_size=320, min_frequency=1, pretokenizer="gpt4")
    text = "Hello, world! 123456789 don't stop.\nNew line here."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_pretokenizer_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="pretokenizer must be"):
        train_byte_bpe(_gpt4_corpus(), vocab_size=320, min_frequency=1, pretokenizer="gpt3")


def test_pretokenizer_gpt4_rejects_combination_with_digit_split() -> None:
    with pytest.raises(ValueError, match="untested combination"):
        train_byte_bpe(
            _gpt4_corpus(),
            vocab_size=320,
            min_frequency=1,
            pretokenizer="gpt4",
            digit_split="no_leading_space",
        )


def test_train_byte_bpe_rejects_unknown_digit_split_mode() -> None:
    with pytest.raises(ValueError, match="digit_split"):
        train_byte_bpe(["abc"], vocab_size=280, min_frequency=1, digit_split="bogus")
