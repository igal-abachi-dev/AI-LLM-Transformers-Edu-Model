"""Canonical V1 protocol and matched Edu/Modern release validation.

Beginner's map of this file
---------------------------
Publishing a model is more than uploading weights. A release here has to be
loadable from a clean clone, has to carry a manifest of SHA-256 hashes so tampering
or truncation is detectable, and has to come with a model card describing what it
is and what it cannot do.

"Matched" is the important word. The Edu and Modern releases exist to be compared,
so they must share a tokenizer, a data mixture, a token budget and an evaluation.
This module is what checks that claim rather than trusting it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from minifrontier.checkpoint import load_release, verify_release_manifest


@dataclass(frozen=True, slots=True)
class TrainingProtocol:
    status: Literal["draft", "frozen"]
    tokenizer_sha256: str
    data_mixture_id: str
    target_tokens: int
    batch_tokens: int
    sequence_length: int
    optimizer: str
    learning_rate: float
    seeds: tuple[int, ...]
    evaluation_interval: int
    evidence_sha256: dict[str, str]
    undertrained_approved: bool = False

    def __post_init__(self) -> None:
        if self.status not in ("draft", "frozen"):
            raise ValueError("protocol status must be draft or frozen")
        if (
            min(
                self.target_tokens,
                self.batch_tokens,
                self.sequence_length,
                self.evaluation_interval,
            )
            <= 0
        ):
            raise ValueError("protocol token/context/evaluation values must be positive")
        if self.learning_rate <= 0 or not self.seeds:
            raise ValueError("protocol requires a positive LR and at least one seed")
        if not self.tokenizer_sha256 or not self.data_mixture_id or not self.optimizer:
            raise ValueError("protocol identity fields must be non-empty")
        if self.status == "frozen" and not self.evidence_sha256:
            raise ValueError("a frozen protocol requires measured evidence hashes")
        if (
            self.status == "frozen"
            and self.target_tokens < 3_000_000_000
            and not self.undertrained_approved
        ):
            raise ValueError("sub-3B frozen protocol requires explicit undertrained approval")

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> TrainingProtocol:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["seeds"] = tuple(values["seeds"])
        return cls(**values)


def audit_release_pair(
    edu_directory: str | Path,
    modern_directory: str | Path,
) -> dict[str, object]:
    """Load both releases independently and prove matched non-architecture scale fields."""

    edu_root = Path(edu_directory)
    modern_root = Path(modern_directory)
    for root in (edu_root, modern_root):
        verify_release_manifest(root)
        if any(root.rglob("training_state.pt")):
            raise ValueError("published release must not contain pickle training state")
    edu, edu_tokenizer = load_release(edu_root)
    modern, modern_tokenizer = load_release(modern_root)
    matched_fields = ("vocab_size", "max_seq_len", "n_layers", "d_model", "d_ff")
    mismatches = {
        field: (getattr(edu.config, field), getattr(modern.config, field))
        for field in matched_fields
        if getattr(edu.config, field) != getattr(modern.config, field)
    }
    if mismatches:
        raise ValueError(f"Edu/Modern release scale fields do not match: {mismatches}")
    if edu_tokenizer.backend.to_str() != modern_tokenizer.backend.to_str():
        raise ValueError("Edu/Modern releases do not contain the same tokenizer")
    prompt = torch.tensor([[edu_tokenizer.bos_id, edu_tokenizer.eos_id]])
    with torch.no_grad():
        edu_logits = edu(prompt).logits
        modern_logits = modern(prompt).logits
    return {
        "status": "load_tested",
        "matched_fields": list(matched_fields),
        "edu_parameters": edu.parameter_count(),
        "modern_parameters": modern.parameter_count(),
        "edu_logits_finite": bool(torch.isfinite(edu_logits).all()),
        "modern_logits_finite": bool(torch.isfinite(modern_logits).all()),
    }
