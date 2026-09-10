import pytest

from minifrontier.packing import pack_best_fit, pack_bos_aligned_crop


def test_pack_best_fit_never_discards_a_token() -> None:
    documents = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    packed = pack_best_fit(documents, sequence_length=5, pad_id=0)
    assert sum(count for _, count in packed) == sum(len(document) for document in documents)


def test_pack_best_fit_does_not_split_documents_that_fit_together_in_one_row() -> None:
    documents = [[6, 7, 8, 9], [1, 2, 3], [4, 5]]
    packed = pack_best_fit(documents, sequence_length=5, pad_id=0)
    # [1, 2, 3] and [4, 5] together fill a row exactly -- both intact, same row.
    assert any(tokens[:5] == [1, 2, 3, 4, 5] for tokens, _ in packed)


def test_pack_best_fit_splits_an_oversized_document_into_full_chunks_plus_remainder() -> None:
    documents = [list(range(12))]  # length 12, sequence_length 5 -> 5 + 5 + 2
    packed = pack_best_fit(documents, sequence_length=5, pad_id=0)
    full_chunk_counts = [count for _, count in packed if count == 5]
    assert len(full_chunk_counts) == 2
    assert sum(count for _, count in packed) == 12


def test_pack_best_fit_pads_a_leftover_row_and_reports_the_real_count() -> None:
    packed = pack_best_fit([[1, 2, 3]], sequence_length=5, pad_id=0)
    assert len(packed) == 1
    tokens, non_padding = packed[0]
    assert tokens == [1, 2, 3, 0, 0]
    assert non_padding == 3


def test_pack_best_fit_rejects_non_positive_sequence_length() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        pack_best_fit([[1]], sequence_length=0, pad_id=0)


def test_pack_bos_aligned_crop_prepends_bos_to_every_row() -> None:
    documents = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    packed = pack_bos_aligned_crop(documents, sequence_length=3, pad_id=0, bos_id=100)
    assert all(tokens[0] == 100 for tokens, _ in packed)


def test_pack_bos_aligned_crop_fits_multiple_whole_documents_per_row_when_possible() -> None:
    documents = [[1, 2], [3, 4]]
    packed = pack_bos_aligned_crop(documents, sequence_length=5, pad_id=0, bos_id=100)
    assert packed == [([100, 1, 2, 3, 4], 5)]


def test_pack_bos_aligned_crop_crops_and_permanently_discards_overflow() -> None:
    documents = [[1, 2], [3, 4, 5, 6, 7], [8, 9]]
    packed = pack_bos_aligned_crop(documents, sequence_length=5, pad_id=0, bos_id=100)
    # Row 1: BOS + doc0 whole (2) + doc1 cropped to fill the remaining 2 slots
    # ([3, 4]) -- doc1's [5, 6, 7] tail is discarded for good, never appearing
    # anywhere in the output. Row 2: BOS + doc2 whole, padded.
    assert packed[0] == ([100, 1, 2, 3, 4], 5)
    assert packed[1] == ([100, 8, 9, 0, 0], 3)
    all_tokens = [token for tokens, _ in packed for token in tokens]
    assert 5 not in all_tokens and 6 not in all_tokens and 7 not in all_tokens
    # Real, measurable loss: total document content preserved (excluding one
    # BOS per row) is strictly less than the sum of input document lengths.
    content_preserved = sum(count - 1 for _, count in packed)
    assert content_preserved < sum(len(document) for document in documents)


def test_pack_bos_aligned_crop_crops_a_too_long_document_to_fill_exactly_one_row() -> None:
    documents = [list(range(10))]  # far longer than any single row can hold
    packed = pack_bos_aligned_crop(documents, sequence_length=5, pad_id=0, bos_id=100)
    assert len(packed) == 1
    assert packed[0] == ([100, 0, 1, 2, 3], 5)


def test_pack_bos_aligned_crop_rejects_sequence_length_too_small_for_bos() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        pack_bos_aligned_crop([[1]], sequence_length=1, pad_id=0, bos_id=100)
