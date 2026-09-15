import json
from argparse import Namespace
from pathlib import Path

import pyarrow.parquet as pq
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


def test_prepare_output_directories_rejects_and_preserves_partial_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "shards"
    parquet_dir = tmp_path / "parquet"
    output_dir.mkdir()
    parquet_dir.mkdir()
    (output_dir / "partial.npy").write_bytes(b"shard")
    (parquet_dir / ".train.parquet.tmp").write_bytes(b"parquet")

    with pytest.raises(FileExistsError, match="--restart-incomplete"):
        prepare_data.prepare_output_directories(
            output_dir,
            parquet_dir,
            restart_incomplete=False,
        )

    assert (output_dir / "partial.npy").read_bytes() == b"shard"
    assert (parquet_dir / ".train.parquet.tmp").read_bytes() == b"parquet"


def test_prepare_output_directories_restarts_both_partial_publications(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "shards"
    parquet_dir = tmp_path / "parquet"
    output_dir.mkdir()
    parquet_dir.mkdir()
    (output_dir / "partial.npy").write_bytes(b"shard")
    (parquet_dir / ".train.parquet.tmp").write_bytes(b"parquet")

    prepare_data.prepare_output_directories(
        output_dir,
        parquet_dir,
        restart_incomplete=True,
    )

    assert not output_dir.exists()
    assert not parquet_dir.exists()


def test_prepare_output_directories_never_deletes_completed_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "shards"
    parquet_dir = tmp_path / "parquet"
    output_dir.mkdir()
    parquet_dir.mkdir()
    (output_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (parquet_dir / "train.parquet").write_bytes(b"published")

    with pytest.raises(FileExistsError, match="refusing to delete completed"):
        prepare_data.prepare_output_directories(
            output_dir,
            parquet_dir,
            restart_incomplete=True,
        )

    assert (output_dir / "metadata.json").exists()
    assert (parquet_dir / "train.parquet").read_bytes() == b"published"


def test_manifest_source_rejects_fineweb_cursor_options(tmp_path: Path) -> None:
    args = source_args(
        manifest=tmp_path / "documents.jsonl",
        source=None,
        limit=1,
    )
    with pytest.raises(ValueError, match="require a streaming --source"):
        prepare_data.document_stream(args)


def test_prepare_data_selects_ebook_markdown_stream(monkeypatch, tmp_path: Path) -> None:
    observed = {}

    def fake_ebook_markdown(directory, **kwargs):
        observed["directory"] = directory
        observed.update(kwargs)
        return iter(("document",))

    monkeypatch.setattr(prepare_data, "iter_ebook_markdown", fake_ebook_markdown)
    args = source_args(
        source="ebook-markdown",
        ebook_directory=tmp_path,
        ebook_license="CC0-1.0",
        ebook_revision="1st-ed",
        ebook_language="Hebrew",
    )
    assert list(prepare_data.document_stream(args)) == ["document"]
    assert observed == {
        "directory": tmp_path,
        "license": "CC0-1.0",
        "revision": "1st-ed",
        "language": "Hebrew",
        "limit": 12,
        "start": 3,
        "shuffle_seed": 17,
    }


def test_ebook_markdown_source_requires_directory() -> None:
    args = source_args(source="ebook-markdown", ebook_directory=None)
    with pytest.raises(ValueError, match="--ebook-directory"):
        prepare_data.document_stream(args)


def test_prepare_data_end_to_end_ebook_source_with_parquet_export(
    monkeypatch, tmp_path: Path, tokenizer_dir: Path
) -> None:
    ebook_directory = tmp_path / "ebooks"
    texts = {
        "book-a": "The quick brown fox jumps over the lazy dog. " * 20,
        "book-b": "MiniFrontier teaches attention from first principles. " * 20,
    }
    for book_id, text in texts.items():
        book_dir = ebook_directory / "md" / book_id
        book_dir.mkdir(parents=True)
        (book_dir / "book.md").write_text(text, encoding="utf-8")
    output_dir = tmp_path / "shards"
    parquet_dir = tmp_path / "parquet"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--source",
            "ebook-markdown",
            "--ebook-directory",
            str(ebook_directory),
            "--tokenizer",
            str(tokenizer_dir),
            "--output",
            str(output_dir),
            "--sequence-length",
            "16",
            "--validation-fraction",
            "0.5",
            "--export-parquet-dir",
            str(parquet_dir),
        ],
    )
    prepare_data.main()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert not (output_dir / ".metadata.tmp").exists()
    assert metadata["source"] == "ebook-markdown"
    assert metadata["export_parquet_dir"] == str(parquet_dir)
    assert metadata["export_parquet_train_rows"] == 1
    assert metadata["export_parquet_validation_rows"] == 1

    train_table = pq.read_table(parquet_dir / "train-00000-of-00001.parquet")
    validation_table = pq.read_table(parquet_dir / "validation-00000-of-00001.parquet")
    assert [row["record_id"] for row in train_table.to_pylist()] == ["book-b"]
    assert [row["record_id"] for row in validation_table.to_pylist()] == ["book-a"]
    assert train_table.to_pylist()[0]["license"] == "Public Domain"
    assert not (parquet_dir / ".train-00000-of-00001.parquet.tmp").exists()
    assert not (parquet_dir / ".validation-00000-of-00001.parquet.tmp").exists()
