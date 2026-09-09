from dataclasses import replace

import pytest

from minifrontier.checkpoint import export_release
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.release import TrainingProtocol, audit_release_pair, verify_release


def test_training_protocol_stays_draft_without_gpu_evidence(tmp_path) -> None:
    draft = TrainingProtocol(
        status="draft",
        tokenizer_sha256="abc",
        data_mixture_id="mixture-v1",
        target_tokens=3_000_000_000,
        batch_tokens=8192,
        sequence_length=2048,
        optimizer="adamw",
        learning_rate=3e-4,
        seeds=(42,),
        evaluation_interval=100,
        evidence_sha256={},
    )
    path = tmp_path / "protocol.json"
    draft.write(path)
    assert TrainingProtocol.read(path) == draft
    with pytest.raises(ValueError, match="evidence"):
        replace(draft, status="frozen")


def test_matched_release_pair_loads_and_rejects_pickle_state(tmp_path, mini_tokenizer) -> None:
    shared = {
        "vocab_size": max(512, mini_tokenizer.vocab_size),
        "max_seq_len": 16,
        "n_layers": 4,
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 96,
    }
    edu = tmp_path / "edu"
    modern = tmp_path / "modern"
    export_release(edu, MiniFrontier(ModelConfig.tiny_edu(**shared)), mini_tokenizer)
    export_release(
        modern,
        MiniFrontier(ModelConfig.tiny_modern(**shared, n_kv_heads=2, attention_impl="sdpa")),
        mini_tokenizer,
    )
    report = audit_release_pair(edu, modern)
    assert report["status"] == "load_tested"
    (edu / "training_state.pt").write_bytes(b"unsafe")
    with pytest.raises(ValueError, match=r"manifest file set|pickle"):
        audit_release_pair(edu, modern)


def test_verify_release_load_tests_one_directory_and_rejects_pickle_state(
    tmp_path, mini_tokenizer
) -> None:
    directory = tmp_path / "release"
    export_release(
        directory,
        MiniFrontier(ModelConfig.tiny_edu(n_layers=1, d_model=16, n_heads=2, d_ff=32)),
        mini_tokenizer,
    )
    report = verify_release(directory)
    assert report["status"] == "load_tested"
    assert report["logits_finite"] is True

    (directory / "training_state.pt").write_bytes(b"unsafe")
    with pytest.raises(ValueError, match=r"manifest file set|pickle"):
        verify_release(directory)


def test_verify_release_load_tests_mtp_heads_when_present(tmp_path, mini_tokenizer) -> None:
    config = ModelConfig.tiny_modern(vocab_size=max(512, mini_tokenizer.vocab_size))
    mtp_heads = MTPHeads(d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=1)
    directory = tmp_path / "release"
    export_release(directory, MiniFrontier(config), mini_tokenizer, mtp_heads=mtp_heads)

    report = verify_release(directory)
    assert report["mtp_heads_present"] is True

    # A corrupted mtp_heads.safetensors must be caught here -- before
    # scripts/export.py's default cleanup deletes the only other copy of
    # those weights -- not discovered later when someone tries speculative
    # decoding against an already-published release.
    (directory / "sha256-manifest.json").unlink()
    (directory / "mtp_heads.safetensors").write_bytes(b"not a real safetensors file")
    from minifrontier.checkpoint import write_release_manifest

    write_release_manifest(directory)
    with pytest.raises(Exception):  # noqa: B017 -- safetensors' own load error, not this project's
        verify_release(directory)
