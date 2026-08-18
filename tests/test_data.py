import json

import pytest

from minifrontier.data import (
    FINEWEB_EDU_CONFIG,
    FINEWEB_EDU_DATASET,
    FINEWEB_EDU_REVISION,
    Document,
    PackedSequence,
    PackedTokenDataset,
    content_sha256,
    filter_and_deduplicate,
    iter_fineweb_edu,
    iter_jsonl_documents,
    pack_documents,
    split_documents,
)


def document(text: str, record_id: str = "1", **kwargs) -> Document:
    return Document.create(
        text,
        source=kwargs.pop("source", "unit-test"),
        revision=kwargs.pop("revision", "v1"),
        license=kwargs.pop("license", "Apache-2.0"),
        language=kwargs.pop("language", "en"),
        record_id=record_id,
        **kwargs,
    )


def test_document_rejects_bad_hash_and_unapproved_code_license() -> None:
    with pytest.raises(ValueError, match="content_hash"):
        Document("hello", "s", "r", "MIT", "en", "1", "bad")
    with pytest.raises(ValueError, match="not approved"):
        document("print('x')", source_type="code", license="proprietary")


def test_jsonl_manifest_round_trip_and_line_error(tmp_path) -> None:
    path = tmp_path / "documents.jsonl"
    item = document("A sufficiently long document for a manifest.")
    path.write_text(item.to_json() + "\n", encoding="utf-8")
    assert list(iter_jsonl_documents(path)) == [item]
    path.write_text(json.dumps({"text": "missing provenance"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":1"):
        list(iter_jsonl_documents(path))


def test_filter_deduplicates_before_split_and_rejects_bad_content() -> None:
    original = document("This document is long enough to survive filtering.", "a")
    duplicate = document("This document is long enough to survive filtering.", "b")
    nul = document("This has enough characters but a null byte \x00 inside.", "c")
    short = document("short", "d")
    excluded = document("This document is explicitly reserved for evaluation.", "e")
    kept = list(
        filter_and_deduplicate(
            [original, duplicate, nul, short, excluded],
            excluded_hashes={excluded.content_hash},
        )
    )
    assert kept == [original]


def test_document_rejects_bad_split_and_non_utf8_text() -> None:
    with pytest.raises(ValueError, match="invalid split"):
        document("valid UTF-8 text", split="holdout")
    with pytest.raises(UnicodeEncodeError):
        content_sha256("unpaired surrogate: \ud800")


def test_hash_split_is_deterministic_and_disjoint() -> None:
    documents = [
        document(f"Document number {index} has unique deterministic text.", str(index))
        for index in range(200)
    ]
    first_train, first_val = split_documents(documents, validation_fraction=0.2)
    second_train, second_val = split_documents(reversed(documents), validation_fraction=0.2)
    assert {item.content_hash for item in first_train} == {
        item.content_hash for item in second_train
    }
    assert {item.content_hash for item in first_val} == {item.content_hash for item in second_val}
    assert {item.content_hash for item in first_train}.isdisjoint(
        {item.content_hash for item in first_val}
    )
    assert first_train and first_val
    assert {item.split for item in first_train} == {"train"}
    assert {item.split for item in first_val} == {"validation"}


def test_packing_loses_no_tokens_and_marks_padded_remainder(mini_tokenizer) -> None:
    documents = [document("alpha beta gamma", "a"), document("delta epsilon", "b")]
    expected = []
    for item in documents:
        expected.extend(mini_tokenizer.encode(item.text, add_eos=True))
    packed = list(
        pack_documents(documents, mini_tokenizer, sequence_length=7, drop_remainder=False)
    )
    flattened = [token for sequence in packed for token in sequence.token_ids]
    assert flattened[: len(expected)] == expected
    assert all(len(sequence.token_ids) == 7 for sequence in packed)
    assert sum(sequence.non_padding_tokens for sequence in packed) == len(expected)
    assert all(token == mini_tokenizer.pad_id for token in flattened[len(expected) :])


def test_fineweb_adapter_is_bounded_resumable_and_preserves_provenance(monkeypatch) -> None:
    rows = [{"text": f"row {index}", "id": str(index), "language": "en"} for index in range(4)]
    request = {}

    def fake_load_dataset(*args, **kwargs):
        request["args"] = args
        request["kwargs"] = kwargs
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_fineweb_edu(start=1, limit=2))
    assert len(result) == 2
    assert result[0].record_id == "1"
    assert result[0].source == FINEWEB_EDU_DATASET
    assert result[0].revision == FINEWEB_EDU_REVISION
    assert result[0].license == "ODC-BY-1.0"
    assert request == {
        "args": (FINEWEB_EDU_DATASET,),
        "kwargs": {
            "name": FINEWEB_EDU_CONFIG,
            "revision": FINEWEB_EDU_REVISION,
            "split": "train",
            "streaming": True,
        },
    }


def test_fineweb_adapter_requests_deterministic_stream_shuffle(monkeypatch) -> None:
    calls = []

    class FakeDataset:
        def shuffle(self, *, seed, buffer_size):
            calls.append((seed, buffer_size))
            return self

        def __iter__(self):
            yield {"text": "row", "id": "0", "language": "en"}

    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: FakeDataset())
    assert len(list(iter_fineweb_edu(limit=1, shuffle_seed=7, shuffle_buffer=32))) == 1
    assert calls == [(7, 32)]


def test_iterable_dataset_worker_sharding_is_deterministic(monkeypatch) -> None:
    sequences = [PackedSequence((index, index), 2) for index in range(6)]
    worker = type("Worker", (), {"id": 1, "num_workers": 2})()
    monkeypatch.setattr("minifrontier.data.get_worker_info", lambda: worker)
    first = [tensor.tolist() for tensor in PackedTokenDataset(sequences)]
    second = [tensor.tolist() for tensor in PackedTokenDataset(sequences)]
    assert first == second == [[1, 1], [3, 3], [5, 5]]
