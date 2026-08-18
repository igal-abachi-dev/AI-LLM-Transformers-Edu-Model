from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_source_archive_includes_ci_and_excludes_generated_or_binary_files(tmp_path) -> None:
    root = tmp_path / "project"
    for relative in (
        ".github/workflows/ci.yml",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "src/example.py",
        "src/__pycache__/example.pyc",
        "artifacts/model.safetensors",
        "docs/reference.pdf",
        "inference/vendor/source.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    output = tmp_path / "release" / "source.zip"
    script = Path(__file__).parents[1] / "scripts" / "build_source_archive.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert ".github/workflows/ci.yml" in names
    assert "src/example.py" in names
    assert not any(
        name.endswith((".pyc", ".pdf", ".safetensors")) or name.startswith("inference/")
        for name in names
    )
