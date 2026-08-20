"""Fail-closed llama.cpp conversion and GGUF quantization orchestration.

Beginner's map of this file
---------------------------
GGUF is llama.cpp's model format, and quantization means storing weights in 4 or 5
bits instead of 16 -- a large model then fits on a laptop, at some cost in quality.

There is a catch this module refuses to paper over: llama.cpp needs to know the
architecture it is loading, and MiniFrontier's hybrid local/global schedule is not
one it implements. So conversion **fails closed** rather than silently producing a
file that loads and generates plausible nonsense. That refusal is the correct
behaviour, not a missing feature.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from minifrontier.ecosystem import run_checked, sha256_file

GGUF_MAGIC = b"GGUF"


def verify_gguf(path: str | Path) -> None:
    source = Path(path)
    if not source.is_file() or source.stat().st_size <= 4:
        raise ValueError(f"GGUF artifact is missing or empty: {source}")
    with source.open("rb") as file:
        if file.read(4) != GGUF_MAGIC:
            raise ValueError(f"artifact does not have GGUF magic: {source}")


def git_revision(directory: str | Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"cannot resolve pinned checkout revision: {completed.stderr.strip()}")
    return completed.stdout.strip()


def verify_llama_cpp_support(checkout: str | Path, *, expected_revision: str) -> dict[str, str]:
    """Require actual converter/metadata/loader/graph markers before conversion can run."""

    root = Path(checkout)
    actual_revision = git_revision(root)
    if actual_revision != expected_revision:
        raise ValueError(
            f"llama.cpp revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    candidates = {
        "gguf_constants": root / "gguf-py" / "gguf" / "constants.py",
        "tensor_mapping": root / "gguf-py" / "gguf" / "tensor_mapping.py",
        "architecture": root / "src" / "llama-arch.cpp",
        "model_registry": root / "src" / "llama-model.cpp",
        "model_declaration": root / "src" / "models" / "models.h",
        "compute_graph": root / "src" / "models" / "minifrontier.cpp",
    }
    missing = [name for name, path in candidates.items() if not path.exists()]
    if missing:
        raise ValueError(f"llama.cpp checkout is missing required files: {missing}")
    absent_markers = [
        name
        for name, path in candidates.items()
        if "minifrontier" not in path.read_text(encoding="utf-8", errors="ignore").casefold()
    ]
    converter_files = [root / "conversion" / "minifrontier.py"]
    converter = next(
        (
            path
            for path in converter_files
            if path.exists()
            and "minifrontier" in path.read_text(encoding="utf-8", errors="ignore").casefold()
        ),
        None,
    )
    if converter is None:
        absent_markers.append("converter")
    if absent_markers:
        raise ValueError(
            "pinned llama.cpp checkout has no complete MiniFrontier architecture support in: "
            f"{sorted(absent_markers)}"
        )
    return {
        "revision": actual_revision,
        **{f"{name}_sha256": sha256_file(path) for name, path in candidates.items()},
        "converter_sha256": sha256_file(converter),
    }


def convert_high_precision_gguf(
    *,
    llama_cpp: str | Path,
    llama_cpp_revision: str,
    transformers_repository: str | Path,
    output: str | Path,
    outtype: Literal["f16", "bf16"],
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    if outtype not in ("f16", "bf16"):
        raise ValueError("high-precision GGUF outtype must be f16 or bf16")
    source = Path(transformers_repository)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "minifrontier":
        raise ValueError("GGUF conversion source must be a MiniFrontier Transformers repository")
    support = verify_llama_cpp_support(llama_cpp, expected_revision=llama_cpp_revision)
    converter = Path(llama_cpp) / "convert_hf_to_gguf.py"
    command = (
        sys.executable,
        str(converter),
        str(source),
        "--outfile",
        str(output),
        "--outtype",
        outtype,
    )
    completed, elapsed = run_checked(command, cwd=llama_cpp, timeout_seconds=timeout_seconds)
    if completed.returncode:
        raise RuntimeError(f"llama.cpp conversion failed: {completed.stderr[-4000:]}")
    verify_gguf(output)
    return {
        "schema_version": 1,
        "status": "high_precision_gguf_created",
        "architecture": "minifrontier",
        "pretends_to_be_llama": False,
        "outtype": outtype,
        "command": list(command),
        "elapsed_seconds": elapsed,
        "llama_cpp": support,
        "source_adapter_sha256": sha256_file(source / "minifrontier_adapter.json"),
        "output_sha256": sha256_file(output),
        "native_parity": "unmeasured",
        "windows_cuda_smoke": "unmeasured",
    }


def validate_calibration_manifest(values: Mapping[str, Any]) -> None:
    required = {
        "source",
        "revision",
        "license",
        "sampling_recipe",
        "tokenizer_sha256",
        "content_manifest_sha256",
        "evaluation_contamination_status",
    }
    missing = sorted(required - set(values))
    if missing or any(not values[key] for key in required):
        raise ValueError(f"calibration manifest is incomplete: {missing}")


def quantize_q4_k_m(
    *,
    llama_quantize: str | Path,
    high_precision_gguf: str | Path,
    output: str | Path,
    llama_cpp_revision: str,
    source_model_revision: str,
    calibration_manifest: Mapping[str, Any] | None,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    verify_gguf(high_precision_gguf)
    if calibration_manifest is not None:
        validate_calibration_manifest(calibration_manifest)
    command = (
        str(llama_quantize),
        str(high_precision_gguf),
        str(output),
        "Q4_K_M",
    )
    completed, elapsed = run_checked(
        command,
        cwd=Path(llama_quantize).resolve().parent,
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode:
        raise RuntimeError(f"llama-quantize failed: {completed.stderr[-4000:]}")
    verify_gguf(output)
    return {
        "schema_version": 1,
        "status": "artifact_created_not_release_validated",
        "quantization": "Q4_K_M",
        "requantized": False,
        "source_precision": "F16_or_BF16",
        "command": list(command),
        "elapsed_seconds": elapsed,
        "llama_cpp_revision": llama_cpp_revision,
        "source_model_revision": source_model_revision,
        "source_sha256": sha256_file(high_precision_gguf),
        "output_sha256": sha256_file(output),
        "output_bytes": Path(output).stat().st_size,
        "calibration": dict(calibration_manifest) if calibration_manifest else "not_used",
        "uncalibrated_comparison_retained": calibration_manifest is None,
        "cli_smoke": "unmeasured",
        "server_smoke": "unmeasured",
        "quality_regression": "unmeasured",
        "publish_ready": False,
    }
