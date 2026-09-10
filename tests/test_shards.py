from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from minifrontier.data import Document
from minifrontier.shards import (
    AdmissionStats,
    CurriculumMixtureProvider,
    DiskDeduplicator,
    MixtureBatchProvider,
    PackedShardDataset,
    ShardBatchProvider,
    ShardManifest,
    ShardRecord,
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


def test_token_shard_writer_rejects_unknown_packing_and_bad_buffer_size(
    tmp_path, mini_tokenizer
) -> None:
    with pytest.raises(ValueError, match="packing mode"):
        TokenShardWriter(tmp_path, mini_tokenizer, sequence_length=4, packing="bogus")
    with pytest.raises(ValueError, match="pack_buffer_documents"):
        TokenShardWriter(tmp_path, mini_tokenizer, sequence_length=4, pack_buffer_documents=0)


def test_token_shard_writer_best_fit_packing_never_discards_a_token(
    tmp_path, mini_tokenizer
) -> None:
    writer = TokenShardWriter(
        tmp_path,
        mini_tokenizer,
        sequence_length=8,
        sequences_per_shard=4,
        packing="best_fit",
        pack_buffer_documents=3,
    )
    documents = [
        make_document(f"document {index} has a few different words in it", str(index))
        for index in range(7)
    ]
    expected_total = sum(
        len(mini_tokenizer.encode(document.text, add_eos=True)) for document in documents
    )
    for document in documents:
        writer.add(document)
    manifest = writer.finalize()
    assert manifest.packing == "best_fit"
    assert manifest.total_non_padding_tokens == expected_total


def test_token_shard_writer_bos_crop_packing_prepends_bos_to_every_row(
    tmp_path, mini_tokenizer
) -> None:
    writer = TokenShardWriter(
        tmp_path,
        mini_tokenizer,
        sequence_length=6,
        sequences_per_shard=4,
        packing="bos_crop",
        pack_buffer_documents=2,
    )
    for index in range(5):
        writer.add(make_document(f"document {index} has several words", str(index)))
    manifest = writer.finalize()
    assert manifest.packing == "bos_crop"
    dataset = PackedShardDataset(tmp_path)
    assert len(dataset) == manifest.total_sequences > 0
    for index in range(len(dataset)):
        tokens, _ = dataset[index]
        assert int(tokens[0]) == mini_tokenizer.bos_id


def test_token_shard_writer_finalize_flushes_a_partial_packed_buffer(
    tmp_path, mini_tokenizer
) -> None:
    """A buffer smaller than pack_buffer_documents at finalize() time must
    still be packed and written, not silently dropped -- unlike ribbon mode's
    drop_remainder, best_fit/bos_crop always flush their final batch."""

    writer = TokenShardWriter(
        tmp_path,
        mini_tokenizer,
        sequence_length=8,
        packing="best_fit",
        pack_buffer_documents=100,
    )
    writer.add(make_document("just one short document here", "0"))
    manifest = writer.finalize(drop_remainder=True)
    assert manifest.total_sequences >= 1  # flushed, not silently dropped
    assert manifest.total_non_padding_tokens > 0


def test_shard_manifest_defaults_packing_to_ribbon_for_backward_compatibility() -> None:
    # A manifest written before the `packing` field existed has no such key.
    old_style = {
        "version": "minifrontier-shards-v2",
        "sequence_length": 4,
        "dtype": "uint16",
        "pad_id": 0,
        "total_sequences": 1,
        "total_non_padding_tokens": 4,
        "shards": [
            {
                "tokens_path": "shard-00000.tokens.npy",
                "counts_path": "shard-00000.counts.npy",
                "sequences": 1,
                "tokens_sha256": "x",
                "counts_sha256": "y",
            }
        ],
    }
    values = dict(old_style)
    values["shards"] = tuple(ShardRecord(**item) for item in values["shards"])
    manifest = ShardManifest(**values)
    assert manifest.packing == "ribbon"


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


def _make_shard_pool(directory, tokenizer, *, prefix: str, count: int) -> PackedShardDataset:
    writer = TokenShardWriter(directory, tokenizer, sequence_length=4, sequences_per_shard=2)
    for index in range(count):
        text = f"{prefix} document {index} has enough tokens"
        writer.add(make_document(text, f"{prefix}-{index}"))
    writer.finalize(drop_remainder=False)
    return PackedShardDataset(directory)


def test_mixture_batch_provider_draws_from_named_sources_at_configured_weights(
    tmp_path, mini_tokenizer
) -> None:
    web = _make_shard_pool(tmp_path / "web", mini_tokenizer, prefix="web", count=12)
    code = _make_shard_pool(tmp_path / "code", mini_tokenizer, prefix="code", count=12)
    providers = {
        "web": ShardBatchProvider(web, batch_size=1, seed=1),
        "code": ShardBatchProvider(code, batch_size=1, seed=2),
    }
    mixture = MixtureBatchProvider(providers, weights={"web": 0.8, "code": 0.2}, seed=42)
    counts = {"web": 0, "code": 0}
    for index in range(200):
        name = mixture._select_source(index)
        counts[name] += 1
        mixture.next_batch()
    # Not an exact 80/20 split (it's a real weighted random draw), but nowhere
    # near 50/50 either -- a real, measurable skew toward the heavier source.
    assert counts["web"] > counts["code"] * 2


def test_mixture_batch_provider_exact_resume_across_all_sources(tmp_path, mini_tokenizer) -> None:
    _make_shard_pool(tmp_path / "web", mini_tokenizer, prefix="web", count=12)
    _make_shard_pool(tmp_path / "code", mini_tokenizer, prefix="code", count=12)

    def build() -> MixtureBatchProvider:
        providers = {
            "web": ShardBatchProvider(PackedShardDataset(tmp_path / "web"), batch_size=1, seed=1),
            "code": ShardBatchProvider(PackedShardDataset(tmp_path / "code"), batch_size=1, seed=2),
        }
        return MixtureBatchProvider(providers, weights={"web": 0.5, "code": 0.5}, seed=7)

    original = build()
    for _ in range(5):
        original.next_batch()
    state = original.state_dict()
    expected = original.next_batch()

    restored = build()
    restored.load_state_dict(state)
    actual = restored.next_batch()
    assert torch.equal(expected.tokens, actual.tokens)


def test_mixture_batch_provider_rejects_changed_weights_on_resume(tmp_path, mini_tokenizer) -> None:
    web = _make_shard_pool(tmp_path / "web", mini_tokenizer, prefix="web", count=8)
    code = _make_shard_pool(tmp_path / "code", mini_tokenizer, prefix="code", count=8)
    providers = {
        "web": ShardBatchProvider(web, batch_size=1, seed=1),
        "code": ShardBatchProvider(code, batch_size=1, seed=2),
    }
    original = MixtureBatchProvider(providers, weights={"web": 0.5, "code": 0.5}, seed=3)
    state = original.state_dict()

    changed_providers = {
        "web": ShardBatchProvider(web, batch_size=1, seed=1),
        "code": ShardBatchProvider(code, batch_size=1, seed=2),
    }
    changed = MixtureBatchProvider(changed_providers, weights={"web": 0.9, "code": 0.1}, seed=3)
    with pytest.raises(ValueError, match="mixture weights"):
        changed.load_state_dict(state)


def test_mixture_batch_provider_rejects_mismatched_source_names() -> None:
    with pytest.raises(ValueError, match="same sources"):
        MixtureBatchProvider({"web": object()}, weights={"code": 1.0})  # type: ignore[arg-type]


def test_mixture_batch_provider_rejects_empty_or_non_positive_weights(
    tmp_path, mini_tokenizer
) -> None:
    web = _make_shard_pool(tmp_path / "web", mini_tokenizer, prefix="web", count=4)
    with pytest.raises(ValueError, match="at least one source"):
        MixtureBatchProvider({}, weights={})
    with pytest.raises(ValueError, match="positive"):
        MixtureBatchProvider({"web": ShardBatchProvider(web, batch_size=1)}, weights={"web": 0.0})


def test_curriculum_provider_switches_weights_at_the_configured_batch_index(
    tmp_path, mini_tokenizer
) -> None:
    web = _make_shard_pool(tmp_path / "web", mini_tokenizer, prefix="web", count=200)
    code = _make_shard_pool(tmp_path / "code", mini_tokenizer, prefix="code", count=200)
    providers = {
        "web": ShardBatchProvider(web, batch_size=1, seed=1),
        "code": ShardBatchProvider(code, batch_size=1, seed=2),
    }
    curriculum = CurriculumMixtureProvider(
        providers,
        stable_weights={"web": 0.9, "code": 0.1},
        decay_weights={"web": 0.1, "code": 0.9},
        decay_phase_start_batch=100,
        seed=42,
    )
    stable_counts = {"web": 0, "code": 0}
    for index in range(100):
        stable_counts[curriculum._select_source(index)] += 1
    decay_counts = {"web": 0, "code": 0}
    for index in range(100, 200):
        decay_counts[curriculum._select_source(index)] += 1
    # Before the boundary: web-heavy. At/after it: code-heavy -- a real,
    # measurable reversal at exactly the configured switch point.
    assert stable_counts["web"] > stable_counts["code"]
    assert decay_counts["code"] > decay_counts["web"]


def test_curriculum_provider_exact_resume_across_the_phase_boundary(
    tmp_path, mini_tokenizer
) -> None:
    _make_shard_pool(tmp_path / "web", mini_tokenizer, prefix="web", count=12)
    _make_shard_pool(tmp_path / "code", mini_tokenizer, prefix="code", count=12)

    def build() -> CurriculumMixtureProvider:
        providers = {
            "web": ShardBatchProvider(PackedShardDataset(tmp_path / "web"), batch_size=1, seed=1),
            "code": ShardBatchProvider(PackedShardDataset(tmp_path / "code"), batch_size=1, seed=2),
        }
        return CurriculumMixtureProvider(
            providers,
            stable_weights={"web": 0.5, "code": 0.5},
            decay_weights={"web": 0.5, "code": 0.5},
            decay_phase_start_batch=3,
            seed=7,
        )

    original = build()
    for _ in range(5):  # crosses the decay_phase_start_batch=3 boundary
        original.next_batch()
    state = original.state_dict()
    expected = original.next_batch()

    restored = build()
    restored.load_state_dict(state)
    actual = restored.next_batch()
    assert torch.equal(expected.tokens, actual.tokens)


def test_curriculum_provider_rejects_changed_weights_or_boundary_on_resume(
    tmp_path, mini_tokenizer
) -> None:
    web = _make_shard_pool(tmp_path / "web", mini_tokenizer, prefix="web", count=8)
    code = _make_shard_pool(tmp_path / "code", mini_tokenizer, prefix="code", count=8)

    def make(**overrides) -> CurriculumMixtureProvider:
        providers = {
            "web": ShardBatchProvider(web, batch_size=1, seed=1),
            "code": ShardBatchProvider(code, batch_size=1, seed=2),
        }
        defaults = dict(
            stable_weights={"web": 0.5, "code": 0.5},
            decay_weights={"web": 0.2, "code": 0.8},
            decay_phase_start_batch=3,
            seed=3,
        )
        defaults.update(overrides)
        return CurriculumMixtureProvider(providers, **defaults)

    original = make()
    state = original.state_dict()

    with pytest.raises(ValueError, match="stable-phase weights"):
        make(stable_weights={"web": 0.9, "code": 0.1}).load_state_dict(state)
    with pytest.raises(ValueError, match="decay-phase weights"):
        make(decay_weights={"web": 0.9, "code": 0.1}).load_state_dict(state)
    with pytest.raises(ValueError, match="decay_phase_start_batch"):
        make(decay_phase_start_batch=10).load_state_dict(state)


def test_curriculum_provider_rejects_mismatched_source_names_and_bad_weights() -> None:
    with pytest.raises(ValueError, match="same sources"):
        CurriculumMixtureProvider(
            {"web": object()},  # type: ignore[arg-type]
            stable_weights={"code": 1.0},
            decay_weights={"web": 1.0},
            decay_phase_start_batch=1,
        )
    with pytest.raises(ValueError, match="at least one source"):
        CurriculumMixtureProvider(
            {}, stable_weights={}, decay_weights={}, decay_phase_start_batch=1
        )


def test_curriculum_provider_rejects_non_positive_weights_and_negative_boundary(
    tmp_path, mini_tokenizer
) -> None:
    web = _make_shard_pool(tmp_path / "web", mini_tokenizer, prefix="web", count=4)
    providers = {"web": ShardBatchProvider(web, batch_size=1)}
    with pytest.raises(ValueError, match="positive"):
        CurriculumMixtureProvider(
            providers,
            stable_weights={"web": 0.0},
            decay_weights={"web": 1.0},
            decay_phase_start_batch=1,
        )
    with pytest.raises(ValueError, match="positive"):
        CurriculumMixtureProvider(
            providers,
            stable_weights={"web": 1.0},
            decay_weights={"web": 0.0},
            decay_phase_start_batch=1,
        )
    with pytest.raises(ValueError, match="decay_phase_start_batch"):
        CurriculumMixtureProvider(
            providers,
            stable_weights={"web": 1.0},
            decay_weights={"web": 1.0},
            decay_phase_start_batch=-1,
        )


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
