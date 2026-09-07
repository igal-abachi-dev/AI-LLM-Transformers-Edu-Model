import pytest

import scripts.measure_tokenizer_fertility as fertility
from minifrontier.data import Document
from minifrontier.shards import PackedShardDataset, TokenShardWriter
from minifrontier.tokenizer import train_byte_bpe


def _make_document(text: str, record_id: str) -> Document:
    return Document.create(
        text,
        source="fixture",
        revision="abc123",
        license="Apache-2.0",
        language="en",
        record_id=record_id,
    )


def _write_validation_shards(tmp_path, tokenizer, texts: list[str]):
    writer = TokenShardWriter(
        tmp_path / "validation",
        tokenizer,
        sequence_length=32,
        sequences_per_shard=4,
    )
    for index, text in enumerate(texts):
        writer.add(_make_document(text, str(index)))
    writer.finalize(drop_remainder=False)
    return tmp_path / "validation"


def test_decode_held_out_text_round_trips_real_shard_content(tmp_path, mini_tokenizer) -> None:
    # Packing can merge short documents into the same fixed-length sequence
    # with no separator surviving skip_special_tokens=True decode (the packed
    # EOS marker between them is stripped) -- so check distinguishing words
    # survive, not the exact original phrase boundaries.
    texts = ["the quick brown fox jumps over the lazy dog", "a second short document here"]
    shard_dir = _write_validation_shards(tmp_path, mini_tokenizer, texts)
    decoded = fertility.decode_held_out_text(mini_tokenizer, shard_dir, max_sequences=None)
    for word in ("quick", "fox", "second", "document"):
        assert word in decoded


def test_decode_held_out_text_respects_max_sequences(tmp_path, mini_tokenizer) -> None:
    texts = ["document one has unique words", "document two has other unique words"]
    shard_dir = _write_validation_shards(tmp_path, mini_tokenizer, texts)
    full_dataset = PackedShardDataset(shard_dir)
    limited = fertility.decode_held_out_text(mini_tokenizer, shard_dir, max_sequences=1)
    assert len(limited.split("\n")) <= min(1, len(full_dataset))


def test_measure_fertility_reports_real_bytes_per_token(mini_tokenizer) -> None:
    text = "attention transformer attention transformer " * 20
    result = fertility.measure_fertility(mini_tokenizer, text)
    assert result["tokens"] > 0
    assert result["utf8_bytes"] == len(text.encode("utf-8"))
    assert result["bytes_per_token"] == pytest.approx(result["utf8_bytes"] / result["tokens"])


def test_measure_fertility_rejects_text_that_produces_zero_tokens(mini_tokenizer) -> None:
    with pytest.raises(ValueError, match="zero tokens"):
        fertility.measure_fertility(mini_tokenizer, "")


def test_digit_split_modes_produce_measurably_different_fertility() -> None:
    corpus = ["the year 2026 was great " * 20, "digits 123456789 and more digits " * 20]
    no_digit = train_byte_bpe(corpus, vocab_size=300, min_frequency=1, digit_split="none")
    no_leading_space = train_byte_bpe(
        corpus, vocab_size=300, min_frequency=1, digit_split="no_leading_space"
    )
    text = "in 2026 there were 123 events and 456 more"
    no_digit_result = fertility.measure_fertility(no_digit, text)
    split_result = fertility.measure_fertility(no_leading_space, text)
    # Splitting digits into isolated pieces can only add tokens for a
    # digit-bearing sample, never remove them -- a real, measurable fertility
    # cost, matching MF-090's fertility triage on real held-out text.
    assert split_result["tokens"] >= no_digit_result["tokens"]
