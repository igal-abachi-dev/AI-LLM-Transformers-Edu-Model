"""Bounded-memory deduplication, immutable token shards, and exact cursor resume.

Beginner's map of this file
---------------------------
Tokenizing the corpus on the fly during training would be slow and, worse,
non-reproducible. So it happens once, ahead of time, and the result is written to
disk as **shards**: plain files of fixed-width integers that training can read
directly.

Three ideas make this more than "write a big file":

* **Memory-mapped.** A shard is read straight from disk as if it were an array in
  memory. A corpus far larger than RAM works with no special handling.
* **Hashed and immutable.** Each shard records the hash of its contents, so a run
  can state exactly which bytes it trained on, and silent corruption is caught
  rather than trained through.
* **Cursor resume.** Training records which shard and which row it was on. A run
  interrupted at hour nine restarts at hour nine, on the next unseen row, rather
  than re-reading data the model has already learned from.

The deduplication here is the heavier, disk-backed kind (near-duplicates as well
as exact), which is why it lives outside the simple streaming filter in
``data.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from minifrontier.data import Document
from minifrontier.tokenizer import MiniFrontierTokenizer
from minifrontier.training import TrainingBatch

PIPELINE_VERSION = "minifrontier-shards-v2"
NEAR_DEDUP_VERSION = "simhash-token-3gram-v1"


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def simhash64(text: str) -> int:
    tokens = re.findall(r"\w+|[^\w\s]", normalized_text(text))
    features = (
        tokens
        if len(tokens) < 3
        else ["\x1f".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    )
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class DiskDeduplicator:
    """SQLite-backed exact/near signature index that retains no document text."""

    def __init__(self, path: str | Path, *, max_hamming_distance: int = 3) -> None:
        if not 0 <= max_hamming_distance <= 8:
            raise ValueError("max_hamming_distance must be in [0, 8]")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_hamming_distance = max_hamming_distance
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS exact(hash TEXT PRIMARY KEY)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS bands(band INTEGER, value INTEGER, signature TEXT)"
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS band_lookup ON bands(band, value)")

    @staticmethod
    def _bands(signature: int) -> list[tuple[int, int]]:
        return [(band, (signature >> (band * 16)) & 0xFFFF) for band in range(4)]

    def classify(self, text: str) -> tuple[str | None, str, int]:
        exact = normalized_sha256(text)
        signature = simhash64(text)
        if self.connection.execute("SELECT 1 FROM exact WHERE hash = ?", (exact,)).fetchone():
            return "exact_duplicate", exact, signature
        candidates: set[int] = set()
        for band, value in self._bands(signature):
            rows = self.connection.execute(
                "SELECT signature FROM bands WHERE band = ? AND value = ?", (band, value)
            )
            candidates.update(int(row[0], 16) for row in rows)
        if any(
            hamming_distance(signature, candidate) <= self.max_hamming_distance
            for candidate in candidates
        ):
            return "near_duplicate", exact, signature
        return None, exact, signature

    def add(self, exact: str, signature: int) -> None:
        self.connection.execute("INSERT INTO exact(hash) VALUES (?)", (exact,))
        self.connection.executemany(
            "INSERT INTO bands(band, value, signature) VALUES (?, ?, ?)",
            [(band, value, f"{signature:016x}") for band, value in self._bands(signature)],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> DiskDeduplicator:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(slots=True)
class AdmissionStats:
    pipeline_version: str = PIPELINE_VERSION
    near_dedup_version: str = NEAR_DEDUP_VERSION
    max_hamming_distance: int = 3
    seen: int = 0
    admitted: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def admit_documents(
    documents: Any,
    deduplicator: DiskDeduplicator,
    *,
    stats: AdmissionStats,
    min_characters: int = 32,
    max_characters: int = 1_000_000,
    evaluation_exact_hashes: set[str] | frozenset[str] = frozenset(),
    evaluation_simhashes: set[int] | frozenset[int] = frozenset(),
) -> Any:
    """Yield admitted documents while retaining only aggregate rejection reasons."""

    for document in documents:
        stats.seen += 1
        if not min_characters <= len(document.text) <= max_characters:
            stats.reject("character_bounds")
            continue
        if not document.text.strip() or "\x00" in document.text:
            stats.reject("malformed")
            continue
        reason, exact, signature = deduplicator.classify(document.text)
        if reason is not None:
            stats.reject(reason)
            continue
        if exact in evaluation_exact_hashes:
            stats.reject("evaluation_exact_overlap")
            continue
        if any(
            hamming_distance(signature, evaluation) <= stats.max_hamming_distance
            for evaluation in evaluation_simhashes
        ):
            stats.reject("evaluation_near_overlap")
            continue
        deduplicator.add(exact, signature)
        stats.admitted += 1
        yield document


@dataclass(frozen=True, slots=True)
class ShardRecord:
    tokens_path: str
    counts_path: str
    sequences: int
    tokens_sha256: str
    counts_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ShardManifest:
    version: str
    sequence_length: int
    dtype: str
    pad_id: int
    total_sequences: int
    total_non_padding_tokens: int
    shards: tuple[ShardRecord, ...]

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> ShardManifest:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["shards"] = tuple(ShardRecord(**item) for item in values["shards"])
        return cls(**values)


class TokenShardWriter:
    """Stream documents into fixed-size uint16 shards without retaining the corpus."""

    def __init__(
        self,
        directory: str | Path,
        tokenizer: MiniFrontierTokenizer,
        *,
        sequence_length: int,
        sequences_per_shard: int = 1024,
    ) -> None:
        if sequence_length < 2 or sequences_per_shard <= 0:
            raise ValueError("invalid shard dimensions")
        if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
            raise ValueError("uint16 shards require a vocabulary no larger than 65536")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.sequences_per_shard = sequences_per_shard
        self.token_buffer: list[int] = []
        self.sequence_buffer: list[list[int]] = []
        self.count_buffer: list[int] = []
        self.records: list[ShardRecord] = []
        self.total_tokens = 0

    def add(self, document: Document) -> None:
        self.token_buffer.extend(self.tokenizer.encode(document.text, add_eos=True))
        while len(self.token_buffer) >= self.sequence_length:
            sequence = self.token_buffer[: self.sequence_length]
            del self.token_buffer[: self.sequence_length]
            self._append_sequence(sequence, self.sequence_length)

    def _append_sequence(self, tokens: list[int], non_padding: int) -> None:
        self.sequence_buffer.append(tokens)
        self.count_buffer.append(non_padding)
        self.total_tokens += non_padding
        if len(self.sequence_buffer) >= self.sequences_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self.sequence_buffer:
            return
        index = len(self.records)
        tokens_final = self.directory / f"shard-{index:05d}.tokens.npy"
        counts_final = self.directory / f"shard-{index:05d}.counts.npy"
        tokens_temporary = self.directory / f".shard-{index:05d}.tokens.tmp"
        counts_temporary = self.directory / f".shard-{index:05d}.counts.tmp"
        with tokens_temporary.open("wb") as file:
            np.save(file, np.asarray(self.sequence_buffer, dtype=np.uint16), allow_pickle=False)
        with counts_temporary.open("wb") as file:
            np.save(file, np.asarray(self.count_buffer, dtype=np.uint32), allow_pickle=False)
        os.replace(tokens_temporary, tokens_final)
        os.replace(counts_temporary, counts_final)
        self.records.append(
            ShardRecord(
                tokens_path=tokens_final.name,
                counts_path=counts_final.name,
                sequences=len(self.sequence_buffer),
                tokens_sha256=_file_sha256(tokens_final),
                counts_sha256=_file_sha256(counts_final),
            )
        )
        self.sequence_buffer.clear()
        self.count_buffer.clear()

    def finalize(self, *, drop_remainder: bool = True) -> ShardManifest:
        if self.token_buffer and not drop_remainder:
            non_padding = len(self.token_buffer)
            padded = self.token_buffer + [self.tokenizer.pad_id] * (
                self.sequence_length - non_padding
            )
            self._append_sequence(padded, non_padding)
        self.token_buffer.clear()
        self._flush()
        manifest = ShardManifest(
            PIPELINE_VERSION,
            self.sequence_length,
            "uint16",
            self.tokenizer.pad_id,
            sum(record.sequences for record in self.records),
            self.total_tokens,
            tuple(self.records),
        )
        temporary = self.directory / ".manifest.tmp"
        manifest.write(temporary)
        os.replace(temporary, self.directory / "manifest.json")
        return manifest


class PackedShardDataset(Dataset[tuple[torch.Tensor, int]]):
    """Windows-spawn-safe dataset with one memory-mapped shard cached per worker."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest = ShardManifest.read(self.directory / "manifest.json")
        self.ends: list[int] = []
        self._cached_shard_index: int | None = None
        self._cached_tokens: np.ndarray | None = None
        self._cached_counts: np.ndarray | None = None
        total = 0
        for record in self.manifest.shards:
            tokens_path = self.directory / record.tokens_path
            counts_path = self.directory / record.counts_path
            if _file_sha256(tokens_path) != record.tokens_sha256:
                raise ValueError(f"shard hash mismatch: {tokens_path}")
            if _file_sha256(counts_path) != record.counts_sha256:
                raise ValueError(f"shard hash mismatch: {counts_path}")
            total += record.sequences
            self.ends.append(total)
        if total != self.manifest.total_sequences:
            raise ValueError("manifest total_sequences does not match shard records")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cached_shard_index"] = None
        state["_cached_tokens"] = None
        state["_cached_counts"] = None
        return state

    def __len__(self) -> int:
        return self.manifest.total_sequences

    def _load_shard(self, shard_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_shard_index != shard_index:
            record = self.manifest.shards[shard_index]
            tokens = np.load(
                self.directory / record.tokens_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            counts = np.load(
                self.directory / record.counts_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if tokens.shape != (record.sequences, self.manifest.sequence_length):
                raise ValueError(f"invalid token shard shape: {record.tokens_path}")
            if counts.shape != (record.sequences,):
                raise ValueError(f"invalid count shard shape: {record.counts_path}")
            self._cached_shard_index = shard_index
            self._cached_tokens = tokens
            self._cached_counts = counts
        assert self._cached_tokens is not None and self._cached_counts is not None
        return self._cached_tokens, self._cached_counts

    def index_for_shard_row(self, shard_index: int, row_index: int) -> int:
        if not 0 <= shard_index < len(self.manifest.shards):
            raise IndexError(shard_index)
        record = self.manifest.shards[shard_index]
        if not 0 <= row_index < record.sequences:
            raise IndexError(row_index)
        previous = self.ends[shard_index - 1] if shard_index else 0
        return previous + row_index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        previous = self.ends[shard_index - 1] if shard_index else 0
        local_index = index - previous
        shard_tokens, shard_counts = self._load_shard(shard_index)
        tokens = torch.from_numpy(np.asarray(shard_tokens[local_index], dtype=np.int64))
        non_padding = int(shard_counts[local_index])
        return tokens, non_padding


class ShardBatchProvider:
    """Deterministically shuffled, exact-resume immutable-shard provider."""

    def __init__(
        self,
        dataset: PackedShardDataset,
        *,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0 or len(dataset) == 0:
            raise ValueError("positive batch size and non-empty dataset are required")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.shard_cursor = 0
        self.row_cursor = 0
        self._shard_order: list[int] = []
        self._row_order: list[int] = []
        self._reset_orders()

    def _rng(self, *, shard_index: int | None = None) -> random.Random:
        label = f"{self.seed}:{self.epoch}:{'shards' if shard_index is None else shard_index}"
        derived = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
        return random.Random(derived)

    def _reset_orders(self) -> None:
        self._shard_order = list(range(len(self.dataset.manifest.shards)))
        if self.shuffle:
            self._rng().shuffle(self._shard_order)
        self._reset_row_order()

    def _reset_row_order(self) -> None:
        shard_index = self._shard_order[self.shard_cursor]
        rows = self.dataset.manifest.shards[shard_index].sequences
        self._row_order = list(range(rows))
        if self.shuffle:
            self._rng(shard_index=shard_index).shuffle(self._row_order)

    @property
    def cursor(self) -> int:
        completed = sum(
            self.dataset.manifest.shards[index].sequences
            for index in self._shard_order[: self.shard_cursor]
        )
        return completed + self.row_cursor

    def _next_index(self) -> int:
        shard_index = self._shard_order[self.shard_cursor]
        row_index = self._row_order[self.row_cursor]
        index = self.dataset.index_for_shard_row(shard_index, row_index)
        self.row_cursor += 1
        if self.row_cursor == len(self._row_order):
            self.row_cursor = 0
            self.shard_cursor += 1
            if self.shard_cursor == len(self._shard_order):
                self.epoch += 1
                self.shard_cursor = 0
                self._reset_orders()
            else:
                self._reset_row_order()
        return index

    def next_batch(self) -> TrainingBatch:
        rows = []
        counts = []
        for _ in range(self.batch_size):
            rows.append(self.dataset[self._next_index()])
        tokens = torch.stack([row[0] for row in rows])
        counts.extend(row[1] for row in rows)
        positions = torch.arange(tokens.shape[1]).unsqueeze(0)
        loss_mask = positions < torch.tensor(counts).unsqueeze(1)
        return TrainingBatch(tokens, loss_mask=loss_mask)

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "version": 2,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "shard_cursor": self.shard_cursor,
            "row_cursor": self.row_cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 2:
            raise ValueError("unsupported shard provider state version")
        if int(state["seed"]) != self.seed or bool(state["shuffle"]) != self.shuffle:
            raise ValueError("shard provider seed/shuffle policy does not match checkpoint")
        epoch = int(state["epoch"])
        shard_cursor = int(state["shard_cursor"])
        row_cursor = int(state["row_cursor"])
        if epoch < 0:
            raise ValueError("invalid shard provider state")
        self.epoch = epoch
        self.shard_cursor = shard_cursor
        self.row_cursor = row_cursor
        if not 0 <= self.shard_cursor < len(self.dataset.manifest.shards):
            raise ValueError("invalid shard provider state")
        self._reset_orders()
        if not 0 <= self.row_cursor < len(self._row_order):
            raise ValueError("invalid shard provider state")
