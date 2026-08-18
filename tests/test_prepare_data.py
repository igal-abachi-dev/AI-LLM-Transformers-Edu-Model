from argparse import Namespace
from pathlib import Path

import pytest

import scripts.prepare_data as prepare_data


def source_args(**overrides) -> Namespace:
    values = {
        "manifest": None,
        "source": "fineweb-edu",
        "limit": 12,
        "start": 3,
        "shuffle_seed": 17,
        "shuffle_buffer": 99,
    }
    values.update(overrides)
    return Namespace(**values)


def test_prepare_data_selects_direct_bounded_fineweb_stream(monkeypatch) -> None:
    observed = {}

    def fake_fineweb(**kwargs):
        observed.update(kwargs)
        return iter(("document",))

    monkeypatch.setattr(prepare_data, "iter_fineweb_edu", fake_fineweb)
    assert list(prepare_data.document_stream(source_args())) == ["document"]
    assert observed == {
        "limit": 12,
        "start": 3,
        "shuffle_seed": 17,
        "shuffle_buffer": 99,
    }


def test_manifest_source_rejects_fineweb_cursor_options(tmp_path: Path) -> None:
    args = source_args(
        manifest=tmp_path / "documents.jsonl",
        source=None,
        limit=1,
    )
    with pytest.raises(ValueError, match="require --source"):
        prepare_data.document_stream(args)
