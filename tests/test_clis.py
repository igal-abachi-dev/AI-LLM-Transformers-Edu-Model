from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script",
    [
        "sample.py",
        "chat.py",
        "export.py",
        "eval.py",
        "smoke_50m.py",
        "compare_optimizers.py",
        "../train/sft.py",
        "eval_sft.py",
        "freeze_protocol.py",
        "audit_release.py",
        "compare_releases.py",
        "smoke_muon.py",
        "smoke_sft.py",
        "build_source_archive.py",
        "preflight_scale.py",
        "assemble_scale_measurement.py",
        "decide_scale.py",
        "export_huggingface.py",
        "smoke_vllm_api.py",
        "create_serving_fixture.py",
        "convert_gguf.py",
        "quantize_gguf.py",
    ],
)
def test_user_facing_cli_help(script: str) -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
