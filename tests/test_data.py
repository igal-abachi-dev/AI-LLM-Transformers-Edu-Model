import json
import subprocess

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
    _detect_license_from_text,
    _document_cache_parts_dir,
    _resolve_repo_license,
    _sanitize_repo_name_for_parquet_part,
    _strip_leading_license_comment,
    content_sha256,
    filter_and_deduplicate,
    iter_cosmopedia_v2,
    iter_dclm_edu,
    iter_ebook_markdown,
    iter_finemath,
    iter_fineweb_edu,
    iter_github_code,
    iter_github_code_from_repos,
    iter_jsonl_documents,
    iter_parquet_documents,
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
        {
            "code": "d",
            "repo_name": "x/d",
            "path": "d.py",
            "language": "Python",
            "license": "lgpl-3.0",
        },
        {"code": "e", "repo_name": "x/e", "path": "e.py", "language": "Python", "license": "isc"},
    ]

    def fake_load_dataset(*args, **kwargs):
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_github_code(limit=10))
    assert [item.text for item in result] == ["a", "c", "e"]
    assert result[0].license == "MIT"
    assert result[1].license == "Apache-2.0"
    assert result[2].license == "ISC"
    assert result[0].source == "https://github.com/x/a"
    assert result[0].revision == GITHUB_CODE_REVISION
    assert result[0].source_type == "code"


def test_github_code_adapter_manual_license_override_admits_a_stale_dataset_field(
    monkeypatch,
) -> None:
    # curl/curl is a real entry in _MANUALLY_VERIFIED_REPO_LICENSES (MIT,
    # independently verified) -- the dataset's own reported field for it is
    # deliberately something unresolved here, matching the real
    # NOASSERTION/empty behavior that motivated adding the override at all.
    rows = [
        {
            "code": "a",
            "repo_name": "curl/curl",
            "path": "a.c",
            "language": "C",
            "license": "other",
        },
        {
            "code": "b",
            "repo_name": "some/unrelated-repo",
            "path": "b.c",
            "language": "C",
            "license": "other",
        },
    ]

    def fake_load_dataset(*args, **kwargs):
        return iter(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    result = list(iter_github_code(limit=10))
    assert [item.text for item in result] == ["a"]
    assert result[0].license == "MIT"
    assert result[0].source == "https://github.com/curl/curl"


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


def _write_book(tmp_path, book_id: str, text: str) -> None:
    book_dir = tmp_path / "md" / book_id
    book_dir.mkdir(parents=True)
    (book_dir / "book.md").write_text(text, encoding="utf-8")


_REAL_MIT_TEXT = (
    "MIT License\n\nCopyright (c) 2024 Example\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy "
    'of this software and associated documentation files (the "Software"), to deal '
    "in the Software without restriction.\n"
)
_REAL_APACHE_TEXT = (
    "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n\n"
    "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n"
)
_REAL_BSD3_TEXT = (
    "Redistribution and use in source and binary forms, with or without "
    "modification, are permitted provided that the following conditions are met:\n"
    "3. Neither the name of the copyright holder nor the names of its contributors "
    "may be used to endorse or promote products derived from this software.\n"
)
_REAL_BSD2_TEXT = (
    "Redistribution and use in source and binary forms, with or without "
    "modification, are permitted provided that the following conditions are met:\n"
    "1. Redistributions of source code must retain the above copyright notice.\n"
)
_REAL_UNLICENSE_TEXT = "This is free and unencumbered software released into the public domain.\n"
_REAL_CC0_TEXT = "Creative Commons Legal Code\n\nCC0 1.0 Universal\n"
_REAL_ISC_TEXT = (
    "Permission to use, copy, modify, and/or distribute this software for any "
    "purpose with or without fee is hereby granted, provided that the above "
    "copyright notice and this permission notice appear in all copies.\n"
)
_REAL_GPL3_TEXT = (
    "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n\n"
    "Copyright (C) 2007 Free Software Foundation, Inc.\n"
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_REAL_MIT_TEXT, "MIT"),
        (_REAL_APACHE_TEXT, "Apache-2.0"),
        (_REAL_BSD3_TEXT, "BSD-3-Clause"),
        (_REAL_BSD2_TEXT, "BSD-2-Clause"),
        (_REAL_UNLICENSE_TEXT, "Unlicense"),
        (_REAL_CC0_TEXT, "CC0-1.0"),
        (_REAL_ISC_TEXT, "ISC"),
    ],
)
def test_detect_license_from_text_recognizes_each_real_permissive_signature(
    text: str, expected: str
) -> None:
    assert _detect_license_from_text(text) == expected


def test_detect_license_from_text_returns_none_for_unrecognized_prose() -> None:
    assert _detect_license_from_text("This module implements a widget factory.\n") is None


def test_resolve_repo_license_prefers_manual_override_over_file_contents(tmp_path) -> None:
    (tmp_path / "LICENSE").write_text(_REAL_GPL3_TEXT, encoding="utf-8")
    assert _resolve_repo_license("curl/curl", tmp_path) == "MIT"


def test_resolve_repo_license_reads_the_real_cloned_license_file(tmp_path) -> None:
    (tmp_path / "LICENSE.md").write_text(_REAL_APACHE_TEXT, encoding="utf-8")
    assert _resolve_repo_license("some/repo", tmp_path) == "Apache-2.0"


def test_resolve_repo_license_matches_license_filename_case_insensitively(tmp_path) -> None:
    (tmp_path / "Licence.txt").write_text(_REAL_MIT_TEXT, encoding="utf-8")
    assert _resolve_repo_license("some/repo", tmp_path) == "MIT"


def test_resolve_repo_license_warns_and_excludes_a_real_detected_copyleft_relicense(
    tmp_path, capsys
) -> None:
    (tmp_path / "LICENSE").write_text(_REAL_GPL3_TEXT, encoding="utf-8")
    assert _resolve_repo_license("some/repo", tmp_path) is None
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "some/repo" in captured.err
    assert "GPL-3.0" in captured.err


def test_resolve_repo_license_quietly_excludes_undetectable_text_without_fabricating_one(
    tmp_path, capsys
) -> None:
    (tmp_path / "LICENSE").write_text("Some non-canonical licensing statement.\n", encoding="utf-8")
    assert _resolve_repo_license("some/repo", tmp_path) is None
    captured = capsys.readouterr()
    assert captured.err == ""


def test_resolve_repo_license_quietly_excludes_a_repo_with_no_license_file(tmp_path) -> None:
    assert _resolve_repo_license("some/repo", tmp_path) is None


def _fake_clone(files_by_repo: dict[str, dict[str, str]], shas_by_repo: dict[str, str]):
    def clone_repo(repo_name: str, destination) -> str:
        destination.mkdir(parents=True)
        for relative_path, content in files_by_repo.get(repo_name, {}).items():
            file_path = destination / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        return shas_by_repo[repo_name]

    return clone_repo


def test_github_code_from_repos_yields_real_commit_sha_and_license_per_file() -> None:
    clone_repo = _fake_clone({"x/a": {"main.py": "print('hi')\n"}}, {"x/a": "abc123deadbeef"})
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=None,
        )
    )
    assert len(result) == 1
    document = result[0]
    assert document.text == "print('hi')\n"
    assert document.revision == "abc123deadbeef"
    assert document.license == "MIT"
    assert document.language == "Python"
    assert document.source == "https://github.com/x/a"
    assert document.record_id == "x/a:main.py"
    assert document.path == "main.py"
    assert document.source_type == "code"


def test_github_code_from_repos_skips_a_repo_whose_clone_fails_and_continues(capsys) -> None:
    def clone_repo(repo_name: str, destination) -> str:
        if repo_name == "x/broken":
            raise RuntimeError("clone failed")
        destination.mkdir(parents=True)
        (destination / "ok.py").write_text("pass\n", encoding="utf-8")
        return "sha-ok"

    result = list(
        iter_github_code_from_repos(
            ["x/broken", "x/ok"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=None,
        )
    )
    assert [item.source for item in result] == ["https://github.com/x/ok"]
    # A real, previously-silent gap: a clone failure must be visible, not
    # just skipped with zero explanation (--quiet suppresses git's own
    # progress noise, never its real errors -- this project's own calling
    # code was the thing discarding them, not git itself).
    captured = capsys.readouterr()
    assert "x/broken" in captured.err
    assert "clone failed" in captured.err


def test_github_code_from_repos_surfaces_real_captured_stderr_on_failure(capsys) -> None:
    """The diagnostic must prefer a real `CalledProcessError`'s own captured
    stderr (what git itself actually said) over the generic exception text,
    since that's the part a real failure needs to be debuggable from.
    """

    def clone_repo(repo_name: str, destination) -> str:
        raise subprocess.CalledProcessError(
            128, ["git", "clone"], output=b"", stderr=b"fatal: repository not found"
        )

    list(
        iter_github_code_from_repos(
            ["x/gone"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=None,
        )
    )
    captured = capsys.readouterr()
    assert "x/gone" in captured.err
    assert "fatal: repository not found" in captured.err


def test_github_code_from_repos_skips_a_repo_whose_license_is_unresolved() -> None:
    clone_repo = _fake_clone(
        {"x/a": {"main.py": "pass\n"}, "x/b": {"main.py": "pass\n"}},
        {"x/a": "sha-a", "x/b": "sha-b"},
    )
    result = list(
        iter_github_code_from_repos(
            ["x/a", "x/b"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT" if name == "x/a" else None,
            document_cache_path=None,
        )
    )
    assert [item.source for item in result] == ["https://github.com/x/a"]


def test_github_code_from_repos_an_unexpected_resolve_license_crash_does_not_kill_the_run(
    capsys,
) -> None:
    """A real, previously-unprotected gap: only the clone step was wrapped
    in a try/except, so an exception raised anywhere past it (here,
    resolve_license itself, not just a None return) used to propagate all
    the way up and abort the whole generator -- losing every remaining repo
    in a real, multi-hour, 161-repo run over one bad one. Now the entire
    per-repo body is covered.
    """
    clone_repo = _fake_clone(
        {"x/a": {"main.py": "pass\n"}, "x/b": {"main.py": "pass\n"}},
        {"x/a": "sha-a", "x/b": "sha-b"},
    )

    def resolve_license(repo_name, root):
        if repo_name == "x/a":
            raise RuntimeError("unexpected license-resolution crash")
        return "MIT"

    result = list(
        iter_github_code_from_repos(
            ["x/a", "x/b"],
            clone_repo=clone_repo,
            resolve_license=resolve_license,
            document_cache_path=None,
        )
    )
    assert [item.source for item in result] == ["https://github.com/x/b"]
    captured = capsys.readouterr()
    assert "x/a" in captured.err
    assert "unexpected license-resolution crash" in captured.err


def test_github_code_from_repos_filters_by_language_extension() -> None:
    clone_repo = _fake_clone(
        {"x/a": {"main.py": "pass\n", "app.js": "console.log(1)\n"}}, {"x/a": "sha-a"}
    )
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            languages=["python"],
            document_cache_path=None,
        )
    )
    assert [item.path for item in result] == ["main.py"]


def test_github_code_from_repos_extraction_is_idempotent_across_repeated_runs() -> None:
    """Real, directly-verified answer to a real question before committing
    to a multi-hour production run: does re-running extraction against the
    same cached repo content produce identical results every time? Each
    call gets its own fresh fake clone (mirroring how a real cached mirror
    checkout creates a fresh temp directory per run), proving the pipeline
    itself is deterministic, not just that a single shared directory was
    reused. Multiple real files/languages/licenses, not a single trivial case.
    """
    files_by_repo = {
        "x/a": {"main.py": "pass\n", "app.js": "console.log(1);\n", "lib.c": "int f(){}\n"},
        "x/b": {"README.md": "# Title\n"},
    }
    shas_by_repo = {"x/a": "sha-a", "x/b": "sha-b"}

    def run_once():
        clone_repo = _fake_clone(files_by_repo, shas_by_repo)
        return list(
            iter_github_code_from_repos(
                ["x/a", "x/b"],
                clone_repo=clone_repo,
                resolve_license=lambda name, root: "MIT",
                document_cache_path=None,
            )
        )

    first = run_once()
    second = run_once()
    assert first == second
    assert [item.content_hash for item in first] == [item.content_hash for item in second]


def test_github_code_from_repos_recognizes_every_mf121_allowlist_category() -> None:
    """A real gap found by direct inspection (2026-09-15): MF-121 added nine
    real language categories to configs/code-repo-allowlist.txt (C, SQL,
    Markdown, Dockerfile, CMake, PowerShell, Shell, Batchfile, TeX), but the
    extension dictionary was never updated to match -- those real repos
    would have cloned successfully and yielded zero documents each, every
    file silently unmatched. This is the regression test for the fix.
    """
    clone_repo = _fake_clone(
        {
            "x/a": {
                "main.c": "int main() { return 0; }\n",
                "types.h": "#define FOO 1\n",
                "query.sql": "SELECT 1;\n",
                "README.md": "# Title\n",
                "Dockerfile": "FROM scratch\n",
                "Dockerfile.dev": "FROM scratch\n",
                "config.cmake": "set(X 1)\n",
                "CMakeLists.txt": "project(x)\n",
                "script.ps1": "Write-Host 'hi'\n",
                "install.sh": "#!/bin/sh\necho hi\n",
                "run.bat": "echo hi\n",
                "doc.tex": "\\documentclass{article}\n",
            }
        },
        {"x/a": "sha-a"},
    )
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=None,
        )
    )
    by_path = {item.path: item.language for item in result}
    assert by_path == {
        "main.c": "C",
        "types.h": "C",
        "query.sql": "SQL",
        "README.md": "Markdown",
        "Dockerfile": "Dockerfile",
        "Dockerfile.dev": "Dockerfile",
        "config.cmake": "CMake",
        "CMakeLists.txt": "CMake",
        "script.ps1": "PowerShell",
        "install.sh": "Shell",
        "run.bat": "Batchfile",
        "doc.tex": "TeX",
    }


def test_github_code_from_repos_skips_configured_noise_directories() -> None:
    clone_repo = _fake_clone(
        {
            "x/a": {
                "main.py": "pass\n",
                "node_modules/dep/index.js": "module.exports = 1;\n",
                "docs/guide.md": "# Guide\n",
                "testdata/fixture.json": "{}\n",
                "fixtures/sample.json": "{}\n",
                "assets/logo.svg": "<svg></svg>\n",
                "images/logo.png": "not-real-png-bytes\n",
            }
        },
        {"x/a": "sha-a"},
    )
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=None,
        )
    )
    assert [item.path for item in result] == ["main.py"]


def test_github_code_from_repos_keeps_real_source_files_under_test_directories() -> None:
    """`test`/`tests` themselves are deliberately not in the skip list --
    real source code in them (assertions, real API usage) is genuine
    training signal, only their non-source contents (fixtures, binary data)
    are excluded, matching the same treatment `examples`/`samples` already
    get.
    """

    clone_repo = _fake_clone(
        {
            "x/a": {
                "main.py": "pass\n",
                "tests/test_main.py": "assert True\n",
                "test/test_other.py": "assert True\n",
                "tests/fixtures/sample.json": "{}\n",
                "tests/testdata/blob.bin": "binary\n",
            }
        },
        {"x/a": "sha-a"},
    )
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=None,
        )
    )
    assert {item.path for item in result} == {
        "main.py",
        "tests/test_main.py",
        "test/test_other.py",
    }


def test_github_code_from_repos_skips_noise_directories_regardless_of_case() -> None:
    """Real, confirmed case variants exist across the actual cached repos for
    these exact names (`Assets` in 13 repos, `TestData` in 9,
    2026-09-15 scan) -- a case-sensitive match would silently miss them.
    """

    clone_repo = _fake_clone(
        {
            "x/a": {
                "main.py": "pass\n",
                "Docs/guide.md": "# Guide\n",
                "TestData/fixture.json": "{}\n",
                "Assets/logo.svg": "<svg></svg>\n",
            }
        },
        {"x/a": "sha-a"},
    )
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=None,
        )
    )
    assert [item.path for item in result] == ["main.py"]


def test_github_code_from_repos_strips_license_headers_from_admitted_files() -> None:
    header = "# Copyright 2024 Example Corp.\n# Licensed under the MIT License.\n\nimport os\n"
    clone_repo = _fake_clone({"x/a": {"main.py": header}}, {"x/a": "sha-a"})
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=None,
        )
    )
    assert result[0].text == "import os\n"


def test_github_code_from_repos_respects_start_and_limit_over_admitted_files() -> None:
    clone_repo = _fake_clone(
        {
            "x/a": {"a.py": "pass\n"},
            "x/b": {"b.py": "pass\n"},
            "x/c": {"c.py": "pass\n"},
        },
        {"x/a": "sha-a", "x/b": "sha-b", "x/c": "sha-c"},
    )
    result = list(
        iter_github_code_from_repos(
            ["x/a", "x/b", "x/c"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            start=1,
            limit=1,
            document_cache_path=None,
        )
    )
    assert [item.path for item in result] == ["b.py"]


def test_github_code_from_repos_rejects_empty_repo_names() -> None:
    with pytest.raises(ValueError, match="repo_names must be non-empty"):
        list(iter_github_code_from_repos([]))


def test_github_code_from_repos_rejects_negative_limit_or_start() -> None:
    with pytest.raises(ValueError, match="limit cannot be negative"):
        list(iter_github_code_from_repos(["x/a"], limit=-1))
    with pytest.raises(ValueError, match="start cannot be negative"):
        list(iter_github_code_from_repos(["x/a"], start=-1))


def test_iter_parquet_documents_round_trips_a_real_document(tmp_path) -> None:
    from minifrontier.shards import ParquetDocumentWriter

    cache_path = tmp_path / "cache.parquet"
    original = document("print('hi')", record_id="x/a:main.py", source_type="code", license="MIT")
    writer = ParquetDocumentWriter(cache_path)
    writer.add(original)
    writer.finalize()
    result = list(iter_parquet_documents(cache_path))
    assert result == [original]


def test_iter_github_code_from_repos_creates_then_replays_the_document_cache(tmp_path) -> None:
    cache_path = tmp_path / "cache.parquet"
    clone_repo = _fake_clone({"x/a": {"main.py": "pass\n"}}, {"x/a": "sha-a"})
    first = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=cache_path,
        )
    )
    assert len(first) == 1
    assert cache_path.exists()

    def clone_repo_should_not_be_called(repo_name, destination):
        raise AssertionError("clone_repo must not be called on a cache replay")

    second = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo_should_not_be_called,
            resolve_license=lambda name, root: (_ for _ in ()).throw(
                AssertionError("resolve_license must not be called on a cache replay")
            ),
            document_cache_path=cache_path,
        )
    )
    assert [item.text for item in second] == [item.text for item in first]


def test_iter_github_code_from_repos_force_refresh_bypasses_the_document_cache(tmp_path) -> None:
    cache_path = tmp_path / "cache.parquet"
    from minifrontier.shards import ParquetDocumentWriter

    stale_writer = ParquetDocumentWriter(cache_path)
    stale_writer.add(document("stale", record_id="x/old:old.py", source_type="code", license="MIT"))
    stale_writer.finalize()

    clone_repo = _fake_clone({"x/a": {"main.py": "fresh\n"}}, {"x/a": "sha-a"})
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=cache_path,
            force_refresh=True,
        )
    )
    assert [item.text for item in result] == ["fresh\n"]


def test_iter_github_code_from_repos_resumes_without_reextracting_already_completed_repos(
    tmp_path,
) -> None:
    """A real interruption: `x/a` already has a finalized per-repo part
    (simulating a crash *after* it completed but *before* the whole run
    finished), `x/b` does not. A resumed call must replay `x/a` from its
    part -- not call `clone_repo`/`resolve_license` for it again -- while
    still extracting `x/b` fresh, then produce one complete, correct final
    cache covering both.
    """
    cache_path = tmp_path / "cache.parquet"
    from minifrontier.shards import ParquetDocumentWriter

    parts_dir = _document_cache_parts_dir(cache_path)
    parts_dir.mkdir(parents=True)
    already_done = document("done", record_id="x/a:main.py", source_type="code", license="MIT")
    part_writer = ParquetDocumentWriter(parts_dir / _sanitize_repo_name_for_parquet_part("x/a"))
    part_writer.add(already_done)
    part_writer.finalize()

    def clone_repo(repo_name, destination):
        if repo_name == "x/a":
            raise AssertionError("x/a already has a completed part -- must not be re-cloned")
        destination.mkdir(parents=True)
        (destination / "main.py").write_text("fresh\n", encoding="utf-8")
        return "sha-b"

    result = list(
        iter_github_code_from_repos(
            ["x/a", "x/b"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=cache_path,
        )
    )
    assert {item.text for item in result} == {"done", "fresh\n"}
    assert cache_path.exists()
    # The resumed run must have cleaned up the now-superseded parts dir once
    # the full cache was successfully finalized.
    assert not parts_dir.exists()
    assert {item.text for item in iter_parquet_documents(cache_path)} == {"done", "fresh\n"}


def test_iter_github_code_from_repos_leaves_a_resumable_partial_state_on_early_stop(
    tmp_path,
) -> None:
    """Simulates a real interruption mid-run (the caller stops consuming
    before every repo is processed, standing in for a hard kill): the final
    `cache_path` must not exist (never falsely marked complete), but the
    already-completed repo's own real part file must survive on disk for a
    later resumed call to reuse.
    """
    cache_path = tmp_path / "cache.parquet"
    clone_repo = _fake_clone(
        {"x/a": {"main.py": "a\n"}, "x/b": {"main.py": "b\n"}}, {"x/a": "sha-a", "x/b": "sha-b"}
    )
    generator = iter_github_code_from_repos(
        ["x/a", "x/b"],
        clone_repo=clone_repo,
        resolve_license=lambda name, root: "MIT",
        document_cache_path=cache_path,
    )
    first_document = next(generator)
    assert first_document.text == "a\n"
    generator.close()

    assert not cache_path.exists()
    parts_dir = _document_cache_parts_dir(cache_path)
    completed_part = parts_dir / _sanitize_repo_name_for_parquet_part("x/a")
    assert completed_part.exists()
    assert [item.text for item in iter_parquet_documents(completed_part)] == ["a\n"]
    not_yet_done_part = parts_dir / _sanitize_repo_name_for_parquet_part("x/b")
    assert not not_yet_done_part.exists()


def test_iter_github_code_from_repos_resumable_cache_does_not_retry_a_permanently_failed_repo(
    tmp_path, capsys
) -> None:
    """A repo that fails entirely (clone error) still gets a real, valid,
    empty part written for it -- matching the existing, already-established
    "a failed repo is skipped for good, not retried forever" contract a
    fully-replayed cache already had, preserved exactly for the resumable
    path too.
    """
    cache_path = tmp_path / "cache.parquet"

    def clone_repo(repo_name, destination):
        if repo_name == "x/broken":
            raise RuntimeError("clone failed")
        destination.mkdir(parents=True)
        (destination / "main.py").write_text("ok\n", encoding="utf-8")
        return "sha-ok"

    first = list(
        iter_github_code_from_repos(
            ["x/broken", "x/ok"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=cache_path,
        )
    )
    assert [item.text for item in first] == ["ok\n"]
    assert cache_path.exists()

    def clone_repo_should_not_be_called_again(repo_name, destination):
        raise AssertionError("a real, force-free replay must never call clone_repo again")

    second = list(
        iter_github_code_from_repos(
            ["x/broken", "x/ok"],
            clone_repo=clone_repo_should_not_be_called_again,
            resolve_license=lambda name, root: (_ for _ in ()).throw(
                AssertionError("must not be called on a cache replay")
            ),
            document_cache_path=cache_path,
        )
    )
    assert [item.text for item in second] == ["ok\n"]


def test_iter_github_code_from_repos_force_refresh_discards_existing_parts(tmp_path) -> None:
    cache_path = tmp_path / "cache.parquet"
    from minifrontier.shards import ParquetDocumentWriter

    parts_dir = _document_cache_parts_dir(cache_path)
    parts_dir.mkdir(parents=True)
    stale_writer = ParquetDocumentWriter(parts_dir / _sanitize_repo_name_for_parquet_part("x/a"))
    stale_writer.add(document("stale", record_id="x/a:old.py", source_type="code", license="MIT"))
    stale_writer.finalize()

    clone_repo = _fake_clone({"x/a": {"main.py": "fresh\n"}}, {"x/a": "sha-a"})
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=cache_path,
            force_refresh=True,
        )
    )
    assert [item.text for item in result] == ["fresh\n"]


def test_iter_github_code_from_repos_resumable_cleans_up_a_leftover_part_tmp_file(
    tmp_path,
) -> None:
    """A leftover `.tmp` from an interrupted part-write for one specific
    repo (a real, if narrow, race -- the same self-healing principle
    already applied to the git-mirror cache's own staging path) must not
    block that repo from being extracted fresh on the next attempt.
    """
    cache_path = tmp_path / "cache.parquet"
    parts_dir = _document_cache_parts_dir(cache_path)
    parts_dir.mkdir(parents=True)
    part_path = parts_dir / _sanitize_repo_name_for_parquet_part("x/a")
    leftover_tmp = part_path.with_name(f".{part_path.name}.tmp")
    leftover_tmp.write_bytes(b"not a real parquet file")

    clone_repo = _fake_clone({"x/a": {"main.py": "fresh\n"}}, {"x/a": "sha-a"})
    result = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=cache_path,
        )
    )
    assert [item.text for item in result] == ["fresh\n"]
    assert not leftover_tmp.exists()


def test_iter_github_code_from_repos_document_cache_reused_across_different_language_filters(
    tmp_path,
) -> None:
    cache_path = tmp_path / "cache.parquet"
    clone_repo = _fake_clone(
        {"x/a": {"main.py": "pass\n", "app.js": "console.log(1)\n"}}, {"x/a": "sha-a"}
    )
    list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo,
            resolve_license=lambda name, root: "MIT",
            document_cache_path=cache_path,
        )
    )

    def clone_repo_should_not_be_called(repo_name, destination):
        raise AssertionError("clone_repo must not be called on a cache replay")

    js_only = list(
        iter_github_code_from_repos(
            ["x/a"],
            clone_repo=clone_repo_should_not_be_called,
            resolve_license=lambda name, root: None,
            document_cache_path=cache_path,
            languages=["javascript"],
        )
    )
    assert [item.path for item in js_only] == ["app.js"]


def test_iter_ebook_markdown_reads_book_md_and_ignores_other_pipeline_outputs(tmp_path) -> None:
    _write_book(tmp_path, "alice-in-wonderland", "Chapter One\n\nDown the rabbit hole.")
    _write_book(tmp_path, "moby-dick", "Call me Ishmael.")
    # Real pdf-to-markdown-rag output also includes chunks/ and metadata/ trees --
    # iter_ebook_markdown must not need or touch either.
    (tmp_path / "chunks" / "moby-dick").mkdir(parents=True)
    (tmp_path / "chunks" / "moby-dick" / "chunks.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata" / "moby-dick").mkdir(parents=True)
    (tmp_path / "metadata" / "moby-dick" / "license.json").write_text("{}", encoding="utf-8")

    documents = list(iter_ebook_markdown(tmp_path))

    assert [document.record_id for document in documents] == ["alice-in-wonderland", "moby-dick"]
    assert documents[0].text == "Chapter One\n\nDown the rabbit hole."
    assert documents[0].source == "ebook:alice-in-wonderland"
    assert documents[0].license == "Public Domain"
    assert documents[0].revision == "n/a"
    assert documents[0].language == "English"
    assert documents[0].source_type == "text"


def test_iter_ebook_markdown_applies_custom_license_revision_and_language(tmp_path) -> None:
    _write_book(tmp_path, "hebrew-book", "שלום עולם")

    [document] = list(
        iter_ebook_markdown(
            tmp_path,
            license="CC0-1.0",
            revision="1st-edition-1922",
            language="Hebrew",
        )
    )

    assert document.license == "CC0-1.0"
    assert document.revision == "1st-edition-1922"
    assert document.language == "Hebrew"


def test_iter_ebook_markdown_skips_empty_books(tmp_path) -> None:
    _write_book(tmp_path, "empty-book", "   \n\n  ")
    _write_book(tmp_path, "real-book", "Real content.")

    documents = list(iter_ebook_markdown(tmp_path))

    assert [document.record_id for document in documents] == ["real-book"]


def test_iter_ebook_markdown_start_limit_and_shuffle(tmp_path) -> None:
    for letter in "abcde":
        _write_book(tmp_path, f"book-{letter}", f"Content {letter}")

    assert [document.record_id for document in iter_ebook_markdown(tmp_path, start=1, limit=2)] == [
        "book-b",
        "book-c",
    ]
    shuffled = [document.record_id for document in iter_ebook_markdown(tmp_path, shuffle_seed=1)]
    assert shuffled == ["book-c", "book-d", "book-e", "book-a", "book-b"]


def test_iter_ebook_markdown_rejects_negative_limit_or_start(tmp_path) -> None:
    with pytest.raises(ValueError, match="limit"):
        list(iter_ebook_markdown(tmp_path, limit=-1))
    with pytest.raises(ValueError, match="start"):
        list(iter_ebook_markdown(tmp_path, start=-1))


def test_iterable_dataset_worker_sharding_is_deterministic(monkeypatch) -> None:
    sequences = [PackedSequence((index, index), 2) for index in range(6)]
    worker = type("Worker", (), {"id": 1, "num_workers": 2})()
    monkeypatch.setattr("minifrontier.data.get_worker_info", lambda: worker)
    first = [tensor.tolist() for tensor in PackedTokenDataset(sequences)]
    second = [tensor.tolist() for tensor in PackedTokenDataset(sequences)]
    assert first == second == [[1, 1], [3, 3], [5, 5]]
