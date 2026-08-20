"""Build a public-review source ZIP without caches, weights, corpora, or vendored snapshots."""

# Packages the source for review or upload. The exclusions are the point: no
# bytecode, no caches, no corpora, no checkpoints, no local artifacts, no
# third-party research bundles.
#
# That is a safety measure as much as a size one. Training directories accumulate
# things that must not be published -- credentials, machine-specific paths,
# licensed data -- and an explicit allow-and-exclude list beats remembering.
# It also refuses to overwrite an existing archive unless you pass --force.

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "inference",
    "tmp",
}
EXCLUDED_SUFFIXES = {
    ".bin",
    ".db",
    ".npy",
    ".npz",
    ".pdf",
    ".pt",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".sqlite",
    ".zip",
}
REQUIRED_FILES = {
    ".github/workflows/ci.yml",
    "README.md",
    "pyproject.toml",
    "uv.lock",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-file-mib", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def collect_source_files(
    root: Path,
    *,
    output: Path,
    max_file_bytes: int,
) -> list[Path]:
    """Return an explicit, sorted public-source allow set and fail closed on large files."""

    root = root.resolve()
    output = output.resolve()
    selected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.resolve() == output:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts[:-1]):
            continue
        if relative.name == ".env" or relative.name.startswith(".env."):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if path.stat().st_size > max_file_bytes:
            raise ValueError(
                f"source file exceeds --max-file-mib and needs an explicit decision: {relative}"
            )
        selected.append(relative)
    relative_names = {path.as_posix() for path in selected}
    missing = REQUIRED_FILES - relative_names
    if missing:
        raise ValueError(f"source archive is missing required files: {sorted(missing)}")
    return selected


def build_source_archive(
    root: Path,
    output: Path,
    *,
    max_file_bytes: int,
    force: bool,
) -> list[Path]:
    root = root.resolve()
    output = output.resolve()
    if output.exists() and not force:
        raise FileExistsError(f"output already exists; pass --force to replace it: {output}")
    selected = collect_source_files(root, output=output, max_file_bytes=max_file_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for relative in selected:
                archive.write(root / relative, relative.as_posix())
        with zipfile.ZipFile(temporary_path) as archive:
            names = set(archive.namelist())
            if not names >= REQUIRED_FILES:
                raise RuntimeError("completed archive failed its required-file verification")
            if any("__pycache__/" in name or name.endswith(".pyc") for name in names):
                raise RuntimeError("completed archive contains Python cache files")
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return selected


def main() -> None:
    args = parse_args()
    if args.max_file_mib <= 0:
        raise ValueError("max-file-mib must be positive")
    files = build_source_archive(
        args.root,
        args.output,
        max_file_bytes=int(args.max_file_mib * 1024 * 1024),
        force=args.force,
    )
    print(f"wrote {args.output} with {len(files)} source files")


if __name__ == "__main__":
    main()
