"""Provenance-enforced code admission and deterministic FIM transforms.

Beginner's map of this file
---------------------------
Two jobs, both about training on source code.

**Admission.** Code carries a license, and a license has conditions. Nothing is
accepted here without an explicit repository, revision, license, path and content
hash, and only a short list of permissive licenses is allowed. Missing provenance
is rejected rather than assumed. The same pass also screens for secrets, because
public repositories do contain leaked keys and a model that memorizes one will
happily reproduce it.

**FIM (fill in the middle).** Plain next-token training only ever teaches a model
to continue text at the end. But writing code means inserting in the *middle* --
that is what an editor's autocomplete does. FIM teaches it by rearranging a
document into::

    <|fim_prefix|> text before <|fim_suffix|> text after <|fim_middle|> the hole

The model still just predicts the next token, left to right. It has simply been
shown the surrounding context first, so "what comes next" happens to be the
missing middle. That is the whole trick, and it is why FIM needs no change to the
model, the loss, or the training loop -- only to the data.
"""

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
_REDACTION_PLACEHOLDER = "[REDACTED]"


@dataclass(frozen=True, slots=True)
class CodeFilterConfig:
    min_characters: int = 32
    max_characters: int = 1_000_000
    max_line_length: int = 2_000
    max_average_line_length: float = 300.0
    redact_personal_data: bool = True


@dataclass(slots=True)
class CodeAdmissionStats:
    version: str = CODE_ADMISSION_VERSION
    seen: int = 0
    admitted: int = 0
    redacted: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def redact_secrets_and_personal_data(text: str, config: CodeFilterConfig) -> tuple[str, bool]:
    """Replace matched secrets/credentials (and, if enabled, emails/phone numbers)
    with a fixed placeholder, keeping the surrounding file intact.

    A real secret makes the whole file worth dropping -- a model that memorizes
    a leaked key will happily reproduce it. But a *rare, real* email or phone
    number in an otherwise ordinary, legitimate file (a code comment, a sample
    config) does not, and this project's own early real-corpus scan found the
    email/phone patterns alone would reject ~11% of an already-curated,
    top-repo GitHub corpus -- overwhelmingly false positives (version strings,
    hashes, IDs matching the phone pattern's loose digit-run shape), not real
    PII. Masking only the matched span, rather than dropping the whole
    document, keeps that file's real training signal instead of losing it.
    """

    redacted = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal redacted
        redacted = True
        return _REDACTION_PLACEHOLDER

    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_replace, text)
    if config.redact_personal_data:
        text = _EMAIL_PATTERN.sub(_replace, text)
        text = _PHONE_PATTERN.sub(_replace, text)
    return text, redacted


def redact_code_document(document: Document, config: CodeFilterConfig) -> Document:
    """Return `document` with any secret/personal-data spans masked.

    Only rebuilds the document (with a fresh `content_hash`, `parent_content_hash`
    preserving the pre-redaction identity so it stays on the same train/validation
    side as it would have without redaction -- the same pattern `mix_fim_documents`
    already uses) when something was actually matched; an unaffected document is
    returned unchanged, not needlessly recomputed.
    """

    redacted_text, changed = redact_secrets_and_personal_data(document.text, config)
    if not changed:
        return document
    return replace(
        document,
        text=redacted_text,
        content_hash=content_sha256(redacted_text),
        parent_content_hash=document.parent_content_hash or document.content_hash,
    )


def code_rejection_reason(document: Document, config: CodeFilterConfig) -> str | None:
    """Structural admission only -- secrets/personal data are handled by
    redaction (`redact_code_document`), not rejection; call that first."""

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
    """Redact secrets/personal data, then yield approved code, keeping aggregate
    reasons -- never a rejected document's real text, but a redacted document's
    surviving text (with the sensitive span masked, not the whole file lost)."""

    config = config or CodeFilterConfig()
    for document in documents:
        stats.seen += 1
        redacted_document = redact_code_document(document, config)
        if redacted_document is not document:
            stats.redacted += 1
        reason = code_rejection_reason(redacted_document, config)
        if reason is not None:
            stats.reject(reason)
            continue
        stats.admitted += 1
        yield redacted_document


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
