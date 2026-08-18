"""Training checkpoints and safe published-model artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model, save_model

from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.reproducibility import capture_rng_state, restore_rng_state
from minifrontier.tokenizer import MiniFrontierTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_release_manifest(directory: str | Path) -> dict[str, str]:
    root = Path(directory)
    manifest = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "sha256-manifest.json"
    }
    (root / "sha256-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_release_manifest(directory: str | Path) -> None:
    root = Path(directory)
    path = root / "sha256-manifest.json"
    if not path.exists():
        raise ValueError("release is missing sha256-manifest.json")
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual_paths = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != path.name
    }
    if set(expected) != actual_paths:
        raise ValueError("release manifest file set does not match directory")
    for relative, digest in expected.items():
        if _sha256(root / relative) != digest:
            raise ValueError(f"release hash mismatch: {relative}")


def _write_flat_toml(config: ModelConfig, path: Path) -> None:
    lines = []
    for key, value in config.to_dict().items():
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = repr(value)
        lines.append(f"{key} = {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_training_checkpoint(
    directory: str | Path,
    model: MiniFrontier,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    data_cursor: Mapping[str, Any] | None = None,
) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    save_model(model, str(target / "model.safetensors"))
    _write_flat_toml(model.config, target / "config.toml")
    (target / "config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "trainer_state.json").write_text(
        json.dumps(dict(trainer_state or {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    local_state = {
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng_state(),
        "data_cursor": dict(data_cursor or {}),
    }
    torch.save(local_state, target / "training_state.pt")


def load_training_checkpoint(
    directory: str | Path,
    model: MiniFrontier,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_rng: bool = True,
    trusted_local_state: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(directory)
    saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if saved_config != model.config.to_dict():
        raise ValueError("checkpoint model configuration does not match the target model")
    load_model(model, str(root / "model.safetensors"), strict=True)
    state_path = root / "training_state.pt"
    local_state: dict[str, Any] = {}
    if state_path.exists():
        if not trusted_local_state:
            raise ValueError(
                "optimizer/RNG state uses pickle; set trusted_local_state=True only for "
                "your checkpoint"
            )
        local_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if optimizer is not None and local_state.get("optimizer") is not None:
            optimizer.load_state_dict(local_state["optimizer"])
        if scheduler is not None and local_state.get("scheduler") is not None:
            scheduler.load_state_dict(local_state["scheduler"])
        if restore_rng and local_state.get("rng") is not None:
            restore_rng_state(local_state["rng"])
    trainer_state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
    return trainer_state, dict(local_state.get("data_cursor", {}))


def export_release(
    directory: str | Path,
    model: MiniFrontier,
    tokenizer: MiniFrontierTokenizer,
    *,
    model_card: str | None = None,
) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    save_model(model, str(target / "model.safetensors"))
    (target / "config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tokenizer.save(target, model_max_length=model.config.max_seq_len)
    template = Path(__file__).parents[2] / "templates" / "chat_template.jinja"
    if template.exists():
        shutil.copyfile(template, target / "chat_template.jinja")
    system_prompt = Path(__file__).parents[2] / "templates" / "system_prompt.md"
    if system_prompt.exists():
        shutil.copyfile(system_prompt, target / "system_prompt.md")
    card = model_card or (
        "# MiniFrontier model\n\n"
        "Development artifact; not a production service or safety-tuned frontier assistant.\n\n"
        "Document training data, hardware, metrics, intended use, limitations, and provenance "
        "before publication. Native MiniFrontier/PyTorch loading is supported; Transformers, "
        "vLLM, and GGUF compatibility require the separate post-V1 adapters.\n"
    )
    (target / "README.md").write_text(card, encoding="utf-8")
    (target / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": tokenizer.bos_id,
                "eos_token_id": tokenizer.eos_id,
                "pad_token_id": tokenizer.pad_id,
                "max_length": model.config.max_seq_len,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_release_manifest(target)


def load_release(
    directory: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MiniFrontier, MiniFrontierTokenizer]:
    root = Path(directory)
    verify_release_manifest(root)
    config = ModelConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    model = MiniFrontier(config).to(device)
    load_model(model, str(root / "model.safetensors"), strict=True, device=str(device))
    model.eval()
    return model, MiniFrontierTokenizer.from_directory(root)
