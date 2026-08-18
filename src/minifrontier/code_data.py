"""Provenance-enforced code admission and deterministic FIM transforms."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any

from minifrontier.data import PERMISSIVE_CODE_LICENSES, Document, content_sha256

CODE_ADMISSION_VERSION = "code-admission-v1"
FIM_TRANSFORM_VERSION = "fim-psm-v1"

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}"),
)
_EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{8,}\d)(?!\d)")
_GENERATED_MARKERS = ("@generated", "generated file", "do not edit", "auto-generated")
_VENDOR_PARTS = {"node_modules", "vendor", "third_party", "dist", "build"}


@dataclass(frozen=True, slots=True)
class CodeFilterConfig:
    min_characters: int = 32
    max_characters: int = 1_000_000
    max_line_length: int = 2_000
    max_average_line_length: float = 300.0
    reject_personal_data: bool = True


@dataclass(slots=True)
class CodeAdmissionStats:
    version: str = CODE_ADMISSION_VERSION
    seen: int = 0
    admitted: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def code_rejection_reason(document: Document, config: CodeFilterConfig) -> str | None:
    if document.source_type != "code":
        return "not_code"
    if not document.path:
        return "missing_path"
    if document.license not in PERMISSIVE_CODE_LICENSES:
        return "license"
    text = document.text
    if not config.min_characters <= len(text) <= config.max_characters:
        return "character_bounds"
    if "\x00" in text or not text.strip():
        return "binary_or_empty"
    path_parts = {part.casefold() for part in re.split(r"[/\\]+", document.path)}
    if path_parts & _VENDOR_PARTS:
        return "vendor_or_generated_path"
    lowered = text[:2_000].casefold()
    if any(marker in lowered for marker in _GENERATED_MARKERS):
        return "generated"
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return "secret_or_credential"
    if config.reject_personal_data and (_EMAIL_PATTERN.search(text) or _PHONE_PATTERN.search(text)):
        return "personal_data"
    lines = text.splitlines() or [text]
    if max(map(len, lines)) > config.max_line_length:
        return "minified_or_malformed"
    if sum(map(len, lines)) / len(lines) > config.max_average_line_length:
        return "minified_or_malformed"
    return None


def filter_code_documents(
    documents: Any,
    *,
    config: CodeFilterConfig | None = None,
    stats: CodeAdmissionStats,
) -> Any:
    """Yield approved code while retaining aggregate reasons, never rejected text."""

    config = config or CodeFilterConfig()
    for document in documents:
        stats.seen += 1
        reason = code_rejection_reason(document, config)
        if reason is not None:
            stats.reject(reason)
            continue
        stats.admitted += 1
        yield document


@dataclass(frozen=True, slots=True)
class FIMTransform:
    prefix: str
    middle: str
    suffix: str
    version: str = FIM_TRANSFORM_VERSION

    def render(self) -> str:
        return f"<|fim_prefix|>{self.prefix}<|fim_suffix|>{self.suffix}<|fim_middle|>{self.middle}"

    def reconstruct(self) -> str:
        return self.prefix + self.middle + self.suffix


def deterministic_fim(text: str, *, seed: int, identity: str) -> FIMTransform:
    if len(text) < 3:
        raise ValueError("FIM requires at least three characters")
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    first = generator.randint(1, len(text) - 2)
    second = generator.randint(first + 1, len(text) - 1)
    return FIMTransform(text[:first], text[first:second], text[second:])


def mix_fim_documents(documents: Any, *, rate: float = 0.15, seed: int) -> Any:
    """Apply deterministic PSM FIM to a reproducible fraction of code documents."""

    if not 0.0 <= rate <= 1.0:
        raise ValueError("FIM rate must be in [0, 1]")
    threshold = int(rate * (1 << 64))
    for document in documents:
        if document.source_type != "code" or len(document.text) < 3:
            yield document
            continue
        selection = int.from_bytes(
            hashlib.sha256(f"fim:{seed}:{document.content_hash}".encode()).digest()[:8],
            "big",
        )
        if selection >= threshold:
            yield document
            continue
        transform = deterministic_fim(document.text, seed=seed, identity=document.content_hash)
        rendered = transform.render()
        yield replace(
            document,
            text=rendered,
            content_hash=content_sha256(rendered),
            record_id=f"{document.record_id}:fim:{FIM_TRANSFORM_VERSION}",
            parent_content_hash=document.parent_content_hash or document.content_hash,
            transform=FIM_TRANSFORM_VERSION,
        )
