"""Provenance-aware document filtering, splitting, tokenization, and packing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import IterableDataset, get_worker_info

from minifrontier.tokenizer import MiniFrontierTokenizer

PERMISSIVE_CODE_LICENSES = frozenset(
    {"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "MIT", "Unlicense"}
)
FINEWEB_EDU_DATASET: Final = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIG: Final = "sample-10BT"
FINEWEB_EDU_REVISION: Final = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Document:
    text: str
    source: str
    revision: str
    license: str
    language: str
    record_id: str
    content_hash: str
    path: str | None = None
    source_type: str = "text"
    split: str | None = None
    parent_content_hash: str | None = None
    transform: str | None = None

    def __post_init__(self) -> None:
        required = {
            "source": self.source,
            "revision": self.revision,
            "license": self.license,
            "language": self.language,
            "record_id": self.record_id,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"missing provenance fields: {', '.join(missing)}")
        expected_hash = content_sha256(self.text)
        if self.content_hash != expected_hash:
            raise ValueError("content_hash does not match UTF-8 document text")
        if self.parent_content_hash is not None and (
            len(self.parent_content_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.parent_content_hash)
        ):
            raise ValueError("parent_content_hash must be a lowercase SHA-256 digest")
        if self.source_type == "code" and self.license not in PERMISSIVE_CODE_LICENSES:
            raise ValueError(f"code license is not approved: {self.license}")
        if self.split not in (None, "train", "validation", "test"):
            raise ValueError(f"invalid split: {self.split}")

    @classmethod
    def create(
        cls,
        text: str,
        *,
        source: str,
        revision: str,
        license: str,
        language: str,
        record_id: str,
        path: str | None = None,
        source_type: str = "text",
        split: str | None = None,
    ) -> Document:
        return cls(
            text=text,
            source=source,
            revision=revision,
            license=license,
            language=language,
            record_id=record_id,
            content_hash=content_sha256(text),
            path=path,
            source_type=source_type,
            split=split,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Document:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PackedSequence:
    token_ids: tuple[int, ...]
    non_padding_tokens: int

    def tensor(self) -> torch.Tensor:
        return torch.tensor(self.token_ids, dtype=torch.long)


def iter_jsonl_documents(path: str | Path) -> Iterator[Document]:
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield Document.from_mapping(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid document at {path}:{line_number}: {error}") from error


def iter_fineweb_edu(
    *,
    limit: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10_000,
) -> Iterator[Document]:
    """Stream a bounded official FineWeb-Edu sample without materializing it."""

    from datasets import load_dataset

    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if start < 0:
        raise ValueError("start cannot be negative")
    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be positive")
    dataset = load_dataset(
        FINEWEB_EDU_DATASET,
        name=FINEWEB_EDU_CONFIG,
        revision=FINEWEB_EDU_REVISION,
        split="train",
        streaming=True,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    emitted = 0
    for index, row in enumerate(dataset):
        if index < start:
            continue
        if limit is not None and emitted >= limit:
            return
        text = str(row["text"])
        yield Document.create(
            text,
            source=FINEWEB_EDU_DATASET,
            revision=FINEWEB_EDU_REVISION,
            license="ODC-BY-1.0",
            language=str(row.get("language", "unknown")),
            record_id=str(row.get("id", index)),
        )
        emitted += 1


def filter_and_deduplicate(
    documents: Iterable[Document],
    *,
    min_characters: int = 32,
    max_characters: int = 1_000_000,
    excluded_hashes: frozenset[str] | set[str] = frozenset(),
) -> Iterator[Document]:
    if min_characters < 0 or max_characters < min_characters:
        raise ValueError("invalid character bounds")
    seen: set[str] = set()
    for document in documents:
        length = len(document.text)
        if length < min_characters or length > max_characters:
            continue
        if not document.text.strip() or "\x00" in document.text:
            continue
        if document.content_hash in seen:
            continue
        if document.content_hash in excluded_hashes:
            continue
        seen.add(document.content_hash)
        yield document


def split_documents(
    documents: Iterable[Document],
    *,
    validation_fraction: float = 0.01,
) -> tuple[list[Document], list[Document]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    train: list[Document] = []
    validation: list[Document] = []
    threshold = int(validation_fraction * 10_000)
    for document in documents:
        bucket = split_bucket(document)
        if bucket < threshold:
            validation.append(replace(document, split="validation"))
        else:
            train.append(replace(document, split="train"))
    return train, validation


def split_bucket(document: Document) -> int:
    """Return a stable split bucket that survives provenance-preserving transforms."""

    identity_hash = document.parent_content_hash or document.content_hash
    return int(identity_hash[:8], 16) % 10_000


def pack_documents(
    documents: Iterable[Document],
    tokenizer: MiniFrontierTokenizer,
    *,
    sequence_length: int,
    drop_remainder: bool = True,
) -> Iterator[PackedSequence]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    buffer: list[int] = []
    for document in documents:
        buffer.extend(tokenizer.encode(document.text, add_eos=True))
        while len(buffer) >= sequence_length:
            yield PackedSequence(tuple(buffer[:sequence_length]), sequence_length)
            del buffer[:sequence_length]
    if buffer and not drop_remainder:
        non_padding = len(buffer)
        buffer.extend([tokenizer.pad_id] * (sequence_length - len(buffer)))
        yield PackedSequence(tuple(buffer), non_padding)


class PackedTokenDataset(IterableDataset[torch.Tensor]):
    """Worker-sharded iterable over already deterministic packed sequences."""

    def __init__(self, sequences: Iterable[PackedSequence]) -> None:
        super().__init__()
        self._sequences = sequences

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        for index, sequence in enumerate(self._sequences):
            if worker is None or index % worker.num_workers == worker.id:
                yield sequence.tensor()
