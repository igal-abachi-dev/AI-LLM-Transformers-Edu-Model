import json

import pytest

from minifrontier.data import (
    COSMOPEDIA_V2_CONFIG,
    DCLM_EDU_DATASET,
    DCLM_EDU_REVISION,
    FINEMATH_DATASET,
    FINEMATH_REVISION,
    FINEWEB_EDU_CONFIG,
    FINEWEB_EDU_DATASET,
    FINEWEB_EDU_REVISION,
    GITHUB_CODE_DATASET,
    GITHUB_CODE_REVISION,
    SMOLLM_CORPUS_DATASET,
    SMOLLM_CORPUS_REVISION,
    Document,
    PackedSequence,
    PackedTokenDataset,
    _strip_leading_license_comment,
    content_sha256,
    filter_and_deduplicate,
    iter_cosmopedia_v2,
    iter_dclm_edu,
    iter_finemath,
    iter_fineweb_edu,
    iter_github_code,
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


def test_dclm_edu_adapter_filters_by_score_and_counts_only_admitted_rows(monkeypatch) -> None:
    # Scores: 5, 1 (rejected), 4, 2 (rejected), 3 -- three real rows pass >=3.
    rows = [
        {"text": "a", "id": "0", "language": "en", "edu_int_score": 5},
        {"text": "b", "id": "1", "language": "en", "edu_int_score": 1},
        {"text": "c", "id": "2", "language": "en", "edu_int_score": 4},
        {"text": "d", "id": "3", "language": "en", "edu_int_score": 2},
        {"text": "e", "id": "4", "language": "en", "edu_int_score": 3},
    ]
    request = {}

    def fake_load_dataset(*args, **kwargs):
        request["args"] = args
        request["kwargs"] = kwargs
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    # start=1 skips the first *admitted* row ("a"), not the first raw row.
    result = list(iter_dclm_edu(min_edu_int_score=3, start=1, limit=10))
    assert [item.text for item in result] == ["c", "e"]
    assert result[0].source == DCLM_EDU_DATASET
    assert result[0].revision == DCLM_EDU_REVISION
    assert result[0].license == "CC-BY-4.0"
    assert request == {
        "args": (DCLM_EDU_DATASET,),
        "kwargs": {"revision": DCLM_EDU_REVISION, "split": "train", "streaming": True},
    }


def test_finemath_adapter_selects_the_requested_config(monkeypatch) -> None:
    rows = [{"text": "row", "id": "0", "language": "en"}]
    request = {}

    def fake_load_dataset(*args, **kwargs):
        request["args"] = args
        request["kwargs"] = kwargs
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_finemath(config="infiwebmath-3plus", limit=1))
    assert len(result) == 1
    assert result[0].source == f"{FINEMATH_DATASET}/infiwebmath-3plus"
    assert result[0].license == "ODC-BY-1.0"
    assert request["kwargs"]["name"] == "infiwebmath-3plus"
    assert request["kwargs"]["revision"] == FINEMATH_REVISION


def test_cosmopedia_v2_adapter_preserves_provenance(monkeypatch) -> None:
    rows = [{"text": "row"}]
    request = {}

    def fake_load_dataset(*args, **kwargs):
        request["args"] = args
        request["kwargs"] = kwargs
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_cosmopedia_v2(limit=1))
    assert len(result) == 1
    assert result[0].source == f"{SMOLLM_CORPUS_DATASET}/{COSMOPEDIA_V2_CONFIG}"
    assert result[0].revision == SMOLLM_CORPUS_REVISION
    assert result[0].license == "ODC-BY-1.0"
    assert request["kwargs"] == {
        "name": COSMOPEDIA_V2_CONFIG,
        "revision": SMOLLM_CORPUS_REVISION,
        "split": "train",
        "streaming": True,
    }


def test_strip_leading_license_comment_removes_real_apache_c_style_header() -> None:
    # Real text captured from a live github-code sample (Apache Software
    # Foundation's own standard header, byte-for-byte identical across
    # thousands of real repositories).
    text = (
        "/*\n"
        " * Licensed to the Apache Software Foundation (ASF) under one\n"
        " * or more contributor license agreements.  See the NOTICE file\n"
        " * distributed with this work for additional information\n"
        " * regarding copyright ownership.  The ASF licenses this file\n"
        ' * to you under the Apache License, Version 2.0 (the "License");\n'
        " */\n"
        "\n"
        "package org.apache.example;\n\npublic class Foo {}\n"
    )
    stripped = _strip_leading_license_comment(text)
    assert stripped == "package org.apache.example;\n\npublic class Foo {}\n"


def test_strip_leading_license_comment_removes_hash_style_header() -> None:
    text = "# Copyright 2024 Example Corp.\n# Licensed under the MIT License.\n\nimport os\n"
    assert _strip_leading_license_comment(text) == "import os\n"


def test_strip_leading_license_comment_removes_docstring_style_header() -> None:
    text = '"""Copyright 2024 Example Corp. Licensed under the MIT License."""\n\nimport os\n'
    assert _strip_leading_license_comment(text) == "import os\n"


def test_strip_leading_license_comment_removes_html_style_header() -> None:
    text = "<!-- Copyright 2024 Example Corp. Licensed under the MIT License. -->\n<html></html>\n"
    assert _strip_leading_license_comment(text) == "<html></html>\n"


def test_strip_leading_license_comment_leaves_non_license_comments_untouched() -> None:
    text = "// This module implements the widget factory.\n\nfunction make() {}\n"
    assert _strip_leading_license_comment(text) == text


def test_strip_leading_license_comment_leaves_code_with_no_leading_comment_untouched() -> None:
    text = "import os\nprint('hello')\n"
    assert _strip_leading_license_comment(text) == text


def test_strip_leading_license_comment_leaves_unterminated_block_comment_untouched() -> None:
    # A real */ never appears -- must not scan unboundedly or crash.
    text = "/* Copyright Example Corp, license text with no closing marker\n" + "x\n" * 5
    assert _strip_leading_license_comment(text) == text


def test_github_code_adapter_strips_license_headers_from_admitted_rows(monkeypatch) -> None:
    header = "# Copyright 2024 Example Corp.\n# Licensed under the MIT License.\n\nimport os\n"
    rows = [
        {
            "code": header,
            "repo_name": "x/a",
            "path": "a.py",
            "language": "Python",
            "license": "mit",
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_github_code(limit=10))
    assert result[0].text == "import os\n"


def test_github_code_adapter_admits_only_permissive_licenses(monkeypatch) -> None:
    rows = [
        {"code": "a", "repo_name": "x/a", "path": "a.py", "language": "Python", "license": "mit"},
        {
            "code": "b",
            "repo_name": "x/b",
            "path": "b.py",
            "language": "Python",
            "license": "gpl-3.0",
        },
        {
            "code": "c",
            "repo_name": "x/c",
            "path": "c.py",
            "language": "Python",
            "license": "apache-2.0",
        },
        {"code": "d", "repo_name": "x/d", "path": "d.py", "language": "Python", "license": "isc"},
    ]

    def fake_load_dataset(*args, **kwargs):
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_github_code(limit=10))
    assert [item.text for item in result] == ["a", "c"]
    assert result[0].license == "MIT"
    assert result[1].license == "Apache-2.0"
    assert result[0].source == "https://github.com/x/a"
    assert result[0].revision == GITHUB_CODE_REVISION
    assert result[0].source_type == "code"


def test_github_code_adapter_filters_by_language_and_repo_allowlist(monkeypatch) -> None:
    rows = [
        {"code": "a", "repo_name": "x/a", "path": "a.py", "language": "Python", "license": "mit"},
        {
            "code": "b",
            "repo_name": "x/b",
            "path": "b.js",
            "language": "JavaScript",
            "license": "mit",
        },
        {"code": "c", "repo_name": "y/c", "path": "c.py", "language": "Python", "license": "mit"},
    ]

    def fake_load_dataset(*args, **kwargs):
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_github_code(languages=["python"], repo_names=["x/a"], limit=10))
    assert [item.text for item in result] == ["a"]


def test_github_code_adapter_start_limit_count_only_admitted_rows(monkeypatch) -> None:
    rows = [
        {"code": "a", "repo_name": "x/a", "path": "a.py", "language": "Python", "license": "mit"},
        {
            "code": "b",
            "repo_name": "x/b",
            "path": "b.py",
            "language": "Python",
            "license": "gpl-3.0",
        },
        {"code": "c", "repo_name": "x/c", "path": "c.py", "language": "Python", "license": "mit"},
        {"code": "d", "repo_name": "x/d", "path": "d.py", "language": "Python", "license": "mit"},
    ]

    def fake_load_dataset(*args, **kwargs):
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_github_code(start=1, limit=10))
    assert [item.text for item in result] == ["c", "d"]


def test_github_code_adapter_requests_streaming_and_trust_remote_code(monkeypatch) -> None:
    request = {}

    def fake_load_dataset(*args, **kwargs):
        request["args"] = args
        request["kwargs"] = kwargs
        return iter([])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    assert list(iter_github_code(limit=1)) == []
    assert request == {
        "args": (GITHUB_CODE_DATASET,),
        "kwargs": {
            "revision": GITHUB_CODE_REVISION,
            "split": "train",
            "streaming": True,
            "trust_remote_code": True,
        },
    }


def test_iterable_dataset_worker_sharding_is_deterministic(monkeypatch) -> None:
    sequences = [PackedSequence((index, index), 2) for index in range(6)]
    worker = type("Worker", (), {"id": 1, "num_workers": 2})()
    monkeypatch.setattr("minifrontier.data.get_worker_info", lambda: worker)
    first = [tensor.tolist() for tensor in PackedTokenDataset(sequences)]
    second = [tensor.tolist() for tensor in PackedTokenDataset(sequences)]
    assert first == second == [[1, 1], [3, 3], [5, 5]]
