from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from minifrontier.data import Document
from minifrontier.shards import (
    AdmissionStats,
    DiskDeduplicator,
    PackedShardDataset,
    ShardBatchProvider,
    TokenShardWriter,
    admit_documents,
    normalized_sha256,
    simhash64,
)


def make_document(text: str, record_id: str) -> Document:
    return Document.create(
        text,
        source="fixture",
        revision="abc123",
        license="Apache-2.0",
        language="en",
        record_id=record_id,
    )


def test_disk_dedup_and_contamination_counters(tmp_path) -> None:
    first = make_document("A sufficiently long document with normalized spacing.", "1")
    normalized_duplicate = make_document(
        "A   sufficiently long document with normalized spacing.", "2"
    )
    contaminated = make_document("A separate evaluation fixture with enough characters.", "3")
    stats = AdmissionStats()
    with DiskDeduplicator(tmp_path / "dedup.sqlite") as dedup:
        admitted = list(
            admit_documents(
                [first, normalized_duplicate, contaminated],
                dedup,
                stats=stats,
                evaluation_simhashes={simhash64(contaminated.text)},
            )
        )
    assert admitted == [first]
    assert stats.admitted == 1
    assert stats.reasons == {"exact_duplicate": 1, "evaluation_near_overlap": 1}
    assert normalized_sha256(first.text) == normalized_sha256(normalized_duplicate.text)


def test_immutable_shards_hashes_and_exact_provider_resume(tmp_path, mini_tokenizer) -> None:
    writer = TokenShardWriter(
        tmp_path / "train",
        mini_tokenizer,
        sequence_length=6,
        sequences_per_shard=2,
    )
    for index in range(8):
        writer.add(
            make_document(f"document {index} with enough repeated words for packing", str(index))
        )
    manifest = writer.finalize(drop_remainder=False)
    assert manifest.total_sequences > 2
    assert len(manifest.shards) > 1
    assert not list((tmp_path / "train").glob("*.tmp"))
    serialized = json.loads((tmp_path / "train" / "manifest.json").read_text())
    assert serialized["version"] == "minifrontier-shards-v2"
    assert all(item["tokens_sha256"] and item["counts_sha256"] for item in serialized["shards"])

    dataset = PackedShardDataset(tmp_path / "train")
    provider = ShardBatchProvider(dataset, batch_size=2)
    first = provider.next_batch()
    state = provider.state_dict()
    expected = provider.next_batch()
    restored = ShardBatchProvider(PackedShardDataset(tmp_path / "train"), batch_size=2)
    restored.load_state_dict(state)
    actual = restored.next_batch()
    assert torch.equal(expected.tokens, actual.tokens)
    assert torch.equal(expected.loss_mask, actual.loss_mask)
    assert first.tokens.dtype == torch.int64
    assert isinstance(dataset._cached_tokens, np.memmap)


def test_shard_shuffle_is_deterministic_complete_and_resume_policy_bound(
    tmp_path, mini_tokenizer
) -> None:
    writer = TokenShardWriter(
        tmp_path,
        mini_tokenizer,
        sequence_length=4,
        sequences_per_shard=2,
    )
    for index in range(12):
        writer.add(make_document(f"document {index} has enough training tokens", str(index)))
    writer.finalize(drop_remainder=False)
    dataset = PackedShardDataset(tmp_path)
    first = ShardBatchProvider(dataset, batch_size=1, seed=91)
    second = ShardBatchProvider(dataset, batch_size=1, seed=91)
    other = ShardBatchProvider(dataset, batch_size=1, seed=92)
    first_order = [first._next_index() for _ in range(len(dataset))]
    second_order = [second._next_index() for _ in range(len(dataset))]
    other_order = [other._next_index() for _ in range(len(dataset))]
    assert first_order == second_order
    assert sorted(first_order) == list(range(len(dataset)))
    assert first_order != other_order

    state = first.state_dict()
    restored = ShardBatchProvider(dataset, batch_size=1, seed=91)
    restored.load_state_dict(state)
    assert restored._next_index() == first._next_index()
    with pytest.raises(ValueError, match="seed/shuffle"):
        ShardBatchProvider(dataset, batch_size=1, seed=90).load_state_dict(state)


def test_shard_hash_corruption_is_rejected(tmp_path, mini_tokenizer) -> None:
    writer = TokenShardWriter(tmp_path, mini_tokenizer, sequence_length=4)
    writer.add(make_document("this text creates at least one packed sequence", "1"))
    manifest = writer.finalize(drop_remainder=False)
    path = tmp_path / manifest.shards[0].tokens_path
    path.write_bytes(path.read_bytes() + b"corrupt")
    try:
        PackedShardDataset(tmp_path)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("corrupt shard was accepted")
