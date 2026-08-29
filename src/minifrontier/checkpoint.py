"""Training checkpoints and safe published-model artifacts.

Beginner's map of this file
---------------------------
Two different things get saved, and confusing them is a classic mistake:

* A **training checkpoint** is a save-game. It holds the weights *plus* the
  optimizer's internal state, the schedule position, and where the data stream had
  got to -- everything needed to resume as if nothing happened. It is large, it is
  private, and it is only ever loaded by this project's own code.
* A **release** is the published model: weights, config, tokenizer, and a card.
  No optimizer state, no data cursor.

Saving the model's weights alone is not enough to resume. AdamW carries two
running averages per weight, so restarting without them makes the optimizer
re-learn its footing and puts a visible bump in the loss curve.

Weights are written with **safetensors**, not ``torch.save``. A PyTorch ``.pt``
file is a Python pickle, and loading a pickle can execute arbitrary code from
whoever produced it -- fine for your own files, unacceptable for a download.
Safetensors is a plain data format that cannot execute anything. Tied embeddings
also need the shared-tensor-aware API here, since the same tensor appears under
two names.
"""

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
    """Write every checkpoint file, then publish them all in one atomic rename.

    A long training run gets killed sometimes -- a machine sleeps, a process
    manager reaps it, power drops. If that happens mid-write with files going
    straight into ``directory``, the result is a checkpoint that *looks* like
    it exists but is actually missing ``config.json``/``trainer_state.json``:
    silently unresumable, and the only way to find out is trying to resume from
    it hours later. Instead, everything is written into a hidden staging
    directory first; only the final ``replace`` (atomic on the same filesystem)
    makes the checkpoint appear at its real path, and only once every file is
    safely on disk. An interrupted save leaves either nothing at ``directory``
    (a fresh checkpoint) or the previous, still-complete one (an overwrite) --
    never a partial one.
    """

    target = Path(directory)
    staging = target.with_name(f".{target.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    save_model(model, str(staging / "model.safetensors"))
    _write_flat_toml(model.config, staging / "config.toml")
    (staging / "config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "trainer_state.json").write_text(
        json.dumps(dict(trainer_state or {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    local_state = {
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng_state(),
        "data_cursor": dict(data_cursor or {}),
    }
    torch.save(local_state, staging / "training_state.pt")
    if target.exists():
        shutil.rmtree(target)
    staging.replace(target)


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
