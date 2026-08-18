from __future__ import annotations

import pytest

from minifrontier.ecosystem import ExternalValidationRecord


def test_external_validation_cannot_mark_unmeasured_evidence_passed() -> None:
    with pytest.raises(ValueError, match="requires parity"):
        ExternalValidationRecord(
            runtime="vllm",
            status="passed",
            runtime_revision="v1",
            model_revision="abc",
            command=("vllm", "serve"),
            hardware="RTX",
            precision="bfloat16",
        )
    with pytest.raises(ValueError, match="unmeasured"):
        ExternalValidationRecord(
            runtime="vllm",
            status="passed",
            runtime_revision="v1",
            model_revision="abc",
            command=("vllm", "serve"),
            hardware="RTX",
            precision="bfloat16",
            parity={"argmax": True},
            metrics={"decode_tokens_per_second": "unmeasured"},
            artifacts={"model": "sha256"},
        )


def test_failed_external_validation_preserves_error() -> None:
    record = ExternalValidationRecord(
        runtime="llama.cpp",
        status="failed",
        runtime_revision="abc",
        model_revision="def",
        command=("llama-cli", "-m", "model.gguf"),
        hardware="Windows RTX",
        precision="F16",
        error="unsupported architecture",
    )
    assert record.error == "unsupported architecture"
