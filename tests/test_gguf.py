from __future__ import annotations

import json
from pathlib import Path

import pytest

from minifrontier.gguf import (
    validate_calibration_manifest,
    verify_gguf,
    verify_llama_cpp_support,
)


def test_gguf_magic_validation(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + b"test")
    verify_gguf(path)
    path.write_bytes(b"nope!")
    with pytest.raises(ValueError, match="GGUF magic"):
        verify_gguf(path)


def test_calibration_manifest_fails_closed() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        validate_calibration_manifest({"source": "x"})
    validate_calibration_manifest(
        {
            "source": "licensed-corpus",
            "revision": "abc",
            "license": "Apache-2.0",
            "sampling_recipe": "seed=42,count=100",
            "tokenizer_sha256": "1" * 64,
            "content_manifest_sha256": "2" * 64,
            "evaluation_contamination_status": "checked-no-match",
        }
    )


def test_llama_cpp_architecture_contract_is_distinct_and_complete() -> None:
    values = json.loads(
        Path("adapters/llama_cpp/minifrontier-architecture.json").read_text(encoding="utf-8")
    )
    assert values["architecture"] == "minifrontier"
    assert values["metadata"]["general.architecture"] == "minifrontier"
    mapping = values["tensor_mapping"]
    assert mapping["model.blocks.{bid}.attention.q_norm.weight"]
    assert mapping["model.blocks.{bid}.attention.k_norm.weight"]
    assert values["graph"]["hybrid_schedule"]
    assert values["tokenizer"]["reserved_ids"]["<|fim_middle|>"] == 8


def test_llama_cpp_support_verifier_matches_current_upstream_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "llama.cpp"
    files = {
        "gguf-py/gguf/constants.py": "MINIFRONTIER\n",
        "gguf-py/gguf/tensor_mapping.py": "minifrontier\n",
        "src/llama-arch.cpp": "minifrontier\n",
        "src/llama-model.cpp": "minifrontier\n",
        "src/models/models.h": "minifrontier\n",
        "src/models/minifrontier.cpp": "minifrontier\n",
        "conversion/minifrontier.py": "minifrontier\n",
    }
    for relative, contents in files.items():
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr("minifrontier.gguf.git_revision", lambda _root: "a" * 40)

    result = verify_llama_cpp_support(checkout, expected_revision="a" * 40)

    assert result["revision"] == "a" * 40
    assert "compute_graph_sha256" in result
    assert "tensor_mapping_sha256" in result


def test_llama_cpp_support_verifier_rejects_missing_compute_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "llama.cpp"
    for relative in (
        "gguf-py/gguf/constants.py",
        "gguf-py/gguf/tensor_mapping.py",
        "src/llama-arch.cpp",
        "src/llama-model.cpp",
        "src/models/models.h",
        "conversion/minifrontier.py",
    ):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("minifrontier\n", encoding="utf-8")
    monkeypatch.setattr("minifrontier.gguf.git_revision", lambda _root: "a" * 40)

    with pytest.raises(ValueError, match="compute_graph"):
        verify_llama_cpp_support(checkout, expected_revision="a" * 40)
