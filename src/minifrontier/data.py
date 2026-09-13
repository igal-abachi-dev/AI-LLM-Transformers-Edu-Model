"""Provenance-aware document filtering, splitting, tokenization, and packing.

Beginner's map of this file
---------------------------
Raw web text is not training data yet. The pipeline that turns one into the other
is, in order:

1. **Stream** documents from a public dataset without downloading terabytes
   (``iter_fineweb_edu``).
2. **Filter and deduplicate** -- drop the too-short, the too-long, the empty, and
   anything already seen. Duplicates are worse than useless: the model memorizes
   them instead of learning the general pattern.
3. **Split** into train and validation *before* anything else touches the text,
   using a hash of the content. Validation only means something if the model has
   genuinely never seen those documents.
4. **Pack** -- tokenize, glue documents end to end with ``<|eos|>`` between them,
   and slice the stream into fixed-length training sequences.

Packing deserves a second look, because it is unintuitive. Rather than padding
every document out to the sequence length -- which would waste most of the
compute on padding -- documents are concatenated into one long ribbon and cut at
fixed intervals. A sequence may therefore contain the end of one document and the
start of the next, separated by ``<|eos|>``, and the model learns from that
boundary too.

Every ``Document`` carries its provenance (source, revision, license, record ID,
content hash) and refuses to be created without it. That is a legal and
reproducibility requirement here, not decoration -- see ``docs/DATA_GOVERNANCE.md``.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import IterableDataset, get_worker_info

from minifrontier.tokenizer import MiniFrontierTokenizer

PERMISSIVE_CODE_LICENSES = frozenset(
    {"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "ISC", "MIT", "Unlicense"}
)
# ISC added 2026-09-12 (MF-121 follow-up): a real, OSI-approved, functionally
# near-identical simplification of MIT/BSD-2-Clause (fewer words, same
# permissions) -- verified before adding, not assumed. Found via two real
# repos (starship/starship, d3/d3) that were otherwise excluded purely
# because ISC wasn't accepted, not because they were actually non-permissive.
FINEWEB_EDU_DATASET: Final = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIG: Final = "sample-10BT"
FINEWEB_EDU_REVISION: Final = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
# MF-095: the other three (of four) real, directly-streamable sources in the
# SmolLM2-modeled mixture proposal, verified against each dataset's own real
# HuggingFace Hub schema before writing any loader (not assumed from a
# similarly-named dataset). Stack-Edu, the fourth (code) component, is NOT
# here -- its own real schema turned out to hold only blob metadata, not code
# text, requiring a separate Software Heritage S3 download pipeline this
# project does not have; see MF-095's backlog status note.
DCLM_EDU_DATASET: Final = "HuggingFaceTB/dclm-edu"
DCLM_EDU_REVISION: Final = "dbad8ad71224482740cd9c9d353591adbf62fe04"
FINEMATH_DATASET: Final = "HuggingFaceTB/finemath"
FINEMATH_REVISION: Final = "e92b25a616738fe95dc186b64dfb19f9c8525594"
# SmollM-Corpus is a multi-config repo; Cosmopedia v2 is one config inside it,
# not a standalone dataset (the standalone "HuggingFaceTB/cosmopedia" repo is
# a different, older dataset -- confirmed before use, not assumed from the name).
SMOLLM_CORPUS_DATASET: Final = "HuggingFaceTB/smollm-corpus"
SMOLLM_CORPUS_REVISION: Final = "3ba9d605774198c5868892d7a8deda78031a781f"
COSMOPEDIA_V2_CONFIG: Final = "cosmopedia-v2"
# MF-095's code component: Stack-Edu was dropped (real schema holds no code
# text, and real content requires a credentialed AWS account -- a real
# project-values decision against ever requiring paid/credentialed cloud
# access to reproduce this pipeline, see MF-095/MF-110's backlog notes).
# github-code is the real, verified, fully ungated replacement: a real `code`
# field, streams with no credentials, and a real per-file `license` field.
GITHUB_CODE_DATASET: Final = "codeparrot/github-code"
GITHUB_CODE_REVISION: Final = "b5661e6b17396364b2bcf8e68977b0d28e1ebd19"
# The dataset's own `languages=`/`licenses=` load_dataset kwargs were tested
# directly against this exact revision under streaming=True and do NOT
# actually filter (verified empirically, not assumed from the dataset card:
# requesting languages=["Python"], licenses=["mit"] still returned
# JavaScript/GPL-2.0 rows) -- iter_github_code filters client-side instead.
# License strings are real, lowercase SPDX-style identifiers (verified via a
# real shuffled sample); normalized to this project's own
# PERMISSIVE_CODE_LICENSES spelling before being handed to Document.create,
# which rejects source_type="code" outside that exact set.
_GITHUB_CODE_PERMISSIVE_LICENSES: Final = {
    "apache-2.0": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "cc0-1.0": "CC0-1.0",
    "isc": "ISC",
    "mit": "MIT",
    "unlicense": "Unlicense",
}
# MF-121 follow-up (2026-09-12): `codeparrot/github-code`'s own per-row
# `license` field is built from GitHub's BigQuery `github_repos` export,
# which historically sources its own license value from the same automated
# detection GitHub's own API exposes -- a real, known source of false
# negatives (a genuinely permissive LICENSE file that the detector doesn't
# recognize as a template match, e.g. a non-canonical file location, an
# unusual preamble, or wording that is functionally but not literally
# identical to the standard text). Thirteen repos hit exactly this (six
# from the original pass, seven more added 2026-09-12 alongside the C#
# allowlist additions -- see `configs/code-repo-allowlist.txt`'s own
# comment for that pass): GitHub's own API reports no resolvable license
# for all thirteen, yet each was independently verified here by reading the
# actual real LICENSE file content directly (not GitHub's automated
# detection) and confirmed genuinely permissive.
# This dict lets a specific, individually-verified repo's real license
# override the dataset's own (possibly stale or undetected) field, rather
# than silently rejecting real, permissively-licensed content -- scoped
# narrowly to exactly these repos, not a general bypass of the license gate.
_MANUALLY_VERIFIED_REPO_LICENSES: Final[dict[str, str]] = {
    # MIT, confirmed via the real COPYING file text (exact clause quoted).
    "curl/curl": "MIT",
    # MIT, confirmed via lua.org (the canonical, authoritative source) --
    # the GitHub mirror simply has no LICENSE file at the path GitHub's
    # detector looks for, not an actual licensing ambiguity.
    "lua/lua": "MIT",
    # BSD-3-Clause, confirmed via the real LICENSE file -- standard 3-clause
    # structure; the non-canonical multi-party copyright header (Facebook,
    # DeepMind, NYU, NEC, IDIAP) is what confuses automated matching, not
    # the substance of the license itself.
    "pytorch/pytorch": "BSD-3-Clause",
    # Real dual license (BSD-3-Clause OR GPL-2.0); the permissive option is
    # explicitly offered by the project itself, confirmed via the real
    # LICENSE file, and is the option used here.
    "facebook/zstd": "BSD-3-Clause",
    # BSD-3-Clause is the real, primary license covering the bulk of the
    # codebase (confirmed via the real LICENSE file); a few minor
    # sub-components carry their own separate (also permissive) licenses.
    "libevent/libevent": "BSD-3-Clause",
    # Apache-2.0, confirmed via the real LICENSE file -- clean, unambiguous.
    "chocolatey/choco": "Apache-2.0",
    # C# allowlist additions (2026-09-12, MF-121 follow-up). Each of the
    # seven entries below reports NOASSERTION via the GitHub API but has a
    # clean, unambiguous, real permissive license confirmed by direct means.
    #
    # Apache-2.0, confirmed via nuget.org's own official, maintainer-set
    # `licenseExpression` field on the real Dapper package (authored by Sam
    # Saffron/Marc Gravell/Nick Craver) -- the repo's own License.txt only
    # points to the license by reference ("licenced under Apache 2.0: URL")
    # rather than embedding it, which is why a prior pass rejected this repo;
    # the nuget metadata is new, genuinely citable, authoritative proof.
    "DapperLib/Dapper": "Apache-2.0",
    # Apache-2.0, confirmed via the real LICENSE.txt file -- a clean, short
    # .NET Foundation copyright notice. This is the legacy pre-ASP.NET-Core
    # SignalR codebase, genuinely distinct from the SignalR implementation
    # bundled inside the already-listed dotnet/aspnetcore.
    "SignalR/SignalR": "Apache-2.0",
    # Apache-2.0, confirmed via the real Licence.txt file (note the British
    # spelling, likely why automated detection misses it) -- a clean Marc
    # Gravell copyright notice using the standard Apache-2.0 boilerplate.
    "protobuf-net/protobuf-net": "Apache-2.0",
    # MIT, confirmed via the real LICENSE file -- the primary, clearly
    # stated project license. The same file also properly attributes two
    # small embedded third-party components under their own separate
    # permissive licenses (BSD-2-Clause for lz4net, Apache-2.0 for .NET
    # Foundation's BufferWriter.cs); unlike the already-rejected lz4/lz4 and
    # meilisearch/meilisearch, every license actually present here is
    # permissive, so this is not a mixed-license case.
    "MessagePack-CSharp/MessagePack-CSharp": "MIT",
    # BSD-3-Clause, confirmed via the real LICENSE file -- standard,
    # unambiguous three-clause structure.
    "cefsharp/CefSharp": "BSD-3-Clause",
    # MIT, confirmed via the real LICENSE.md file -- clean, standard text.
    "OpenTK/OpenTK": "MIT",
    # MIT, confirmed via the real LICENSE file -- a clean Auth0, Inc.
    # copyright notice followed by an auto-generated third-party
    # dependency-license inventory that is not part of, and does not
    # change, the repo's own license.
    "auth0/auth0-aspnetcore-authentication": "MIT",
}


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Document:
    """One piece of training text plus the paperwork that says where it came from.

    ``content_hash`` is the SHA-256 of ``text`` and doubles as the deduplication
    key and the split key. ``parent_content_hash`` points back at the original
    when a document has been transformed (FIM rewriting, for example) so the
    transformed version stays on the same side of the train/validation split as
    its parent.
    """

    text: str
    source: str
    revision: str
    license: str
    language: str
    record_id: str
    content_hash: str
    path: str | None = None
    source_type: str = "text"
    split: str | None = None
    parent_content_hash: str | None = None
    transform: str | None = None

    def __post_init__(self) -> None:
        # Provenance is validated at construction, so an unlabelled document simply
        # cannot exist further down the pipeline.
        required = {
            "source": self.source,
            "revision": self.revision,
            "license": self.license,
            "language": self.language,
            "record_id": self.record_id,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"missing provenance fields: {', '.join(missing)}")
        expected_hash = content_sha256(self.text)
        if self.content_hash != expected_hash:
            raise ValueError("content_hash does not match UTF-8 document text")
        if self.parent_content_hash is not None and (
            len(self.parent_content_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.parent_content_hash)
        ):
            raise ValueError("parent_content_hash must be a lowercase SHA-256 digest")
        if self.source_type == "code" and self.license not in PERMISSIVE_CODE_LICENSES:
            raise ValueError(f"code license is not approved: {self.license}")
        if self.split not in (None, "train", "validation", "test"):
            raise ValueError(f"invalid split: {self.split}")

    @classmethod
    def create(
        cls,
        text: str,
        *,
        source: str,
        revision: str,
        license: str,
        language: str,
        record_id: str,
        path: str | None = None,
        source_type: str = "text",
        split: str | None = None,
    ) -> Document:
        return cls(
            text=text,
            source=source,
            revision=revision,
            license=license,
            language=language,
            record_id=record_id,
            content_hash=content_sha256(text),
            path=path,
            source_type=source_type,
            split=split,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Document:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PackedSequence:
    """One fixed-length training example, ready to become a row of a batch.

    ``non_padding_tokens`` records how much of it is real, which matters only for
    the final remainder sequence when padding was allowed.
    """

    token_ids: tuple[int, ...]
    non_padding_tokens: int

    def tensor(self) -> torch.Tensor:
        return torch.tensor(self.token_ids, dtype=torch.long)


def iter_jsonl_documents(path: str | Path) -> Iterator[Document]:
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield Document.from_mapping(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid document at {path}:{line_number}: {error}") from error


def iter_fineweb_edu(
    *,
    limit: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10_000,
) -> Iterator[Document]:
    """Stream a bounded official FineWeb-Edu sample without materializing it.

    ``streaming=True`` pulls records over the network on demand rather than
    downloading the dataset first. The pinned ``revision`` matters: "FineWeb-Edu"
    is a moving target, and a run record that does not name the exact revision
    cannot be reproduced later.
    """

    from datasets import load_dataset

    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if start < 0:
        raise ValueError("start cannot be negative")
    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be positive")
    dataset = load_dataset(
        FINEWEB_EDU_DATASET,
        name=FINEWEB_EDU_CONFIG,
        revision=FINEWEB_EDU_REVISION,
        split="train",
        streaming=True,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    emitted = 0
    for index, row in enumerate(dataset):
        if index < start:
            continue
        if limit is not None and emitted >= limit:
            return
        text = str(row["text"])
        yield Document.create(
            text,
            source=FINEWEB_EDU_DATASET,
            revision=FINEWEB_EDU_REVISION,
            license="ODC-BY-1.0",
            language=str(row.get("language", "unknown")),
            record_id=str(row.get("id", index)),
        )
        emitted += 1


def iter_dclm_edu(
    *,
    min_edu_int_score: int = 3,
    limit: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10_000,
) -> Iterator[Document]:
    """Stream DCLM-Edu, filtered to `edu_int_score >= min_edu_int_score`.

    The real field name and range (`edu_int_score`, an int64 in 2-5) were
    verified against the dataset's own real HuggingFace schema, not assumed
    from FineWeb-Edu's differently-named quality field. `min_edu_int_score=3`
    is the SmolLM2-modeled mixture's own proposed cutoff (MF-094's status
    note). Unlike `iter_fineweb_edu` (which has no filter of its own, so
    `start`/`limit` count raw stream position), `start`/`limit` here count
    only documents that already passed the score filter -- a caller asking
    for `limit=10_000` gets exactly 10,000 real usable documents, not some
    unpredictable smaller number depending how many of the first N raw rows
    happened to score high enough. A deliberate, documented divergence from
    the raw-position convention, not an oversight.
    """

    from datasets import load_dataset

    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if start < 0:
        raise ValueError("start cannot be negative")
    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be positive")
    dataset = load_dataset(
        DCLM_EDU_DATASET,
        revision=DCLM_EDU_REVISION,
        split="train",
        streaming=True,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    emitted = 0
    admitted_index = 0
    for row in dataset:
        if int(row["edu_int_score"]) < min_edu_int_score:
            continue
        if admitted_index < start:
            admitted_index += 1
            continue
        if limit is not None and emitted >= limit:
            return
        yield Document.create(
            str(row["text"]),
            source=DCLM_EDU_DATASET,
            revision=DCLM_EDU_REVISION,
            license="CC-BY-4.0",
            language=str(row.get("language", "unknown")),
            record_id=str(row.get("id", admitted_index)),
        )
        admitted_index += 1
        emitted += 1


def iter_finemath(
    *,
    config: str = "finemath-4plus",
    limit: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10_000,
) -> Iterator[Document]:
    """Stream FineMath. `config` picks which of its four real subsets to use.

    `finemath-4plus` (the stricter of the two FineMath thresholds) is this
    project's own real, recorded default -- not the paper/dataset's own
    "primary" recommendation, since none is stated; a real decision made
    here, not silently assumed. The other three real configs
    (`finemath-3plus`, `infiwebmath-3plus`, `infiwebmath-4plus`) remain
    available via this same parameter.
    """

    from datasets import load_dataset

    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if start < 0:
        raise ValueError("start cannot be negative")
    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be positive")
    dataset = load_dataset(
        FINEMATH_DATASET,
        name=config,
        revision=FINEMATH_REVISION,
        split="train",
        streaming=True,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    emitted = 0
    for index, row in enumerate(dataset):
        if index < start:
            continue
        if limit is not None and emitted >= limit:
            return
        yield Document.create(
            str(row["text"]),
            source=f"{FINEMATH_DATASET}/{config}",
            revision=FINEMATH_REVISION,
            license="ODC-BY-1.0",
            language=str(row.get("language", "en")),
            record_id=str(row.get("id", index)),
        )
        emitted += 1


def iter_cosmopedia_v2(
    *,
    limit: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10_000,
) -> Iterator[Document]:
    """Stream Cosmopedia v2 (fully synthetic; no quality-score field to filter on)."""

    from datasets import load_dataset

    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if start < 0:
        raise ValueError("start cannot be negative")
    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be positive")
    dataset = load_dataset(
        SMOLLM_CORPUS_DATASET,
        name=COSMOPEDIA_V2_CONFIG,
        revision=SMOLLM_CORPUS_REVISION,
        split="train",
        streaming=True,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    emitted = 0
    for index, row in enumerate(dataset):
        if index < start:
            continue
        if limit is not None and emitted >= limit:
            return
        yield Document.create(
            str(row["text"]),
            source=f"{SMOLLM_CORPUS_DATASET}/{COSMOPEDIA_V2_CONFIG}",
            revision=SMOLLM_CORPUS_REVISION,
            license="ODC-BY-1.0",
            language="en",
            record_id=str(index),
        )
        emitted += 1


_LICENSE_SIGNAL_PATTERN = re.compile(
    r"copyright|licensed under|license\b|spdx-license-identifier|"
    r"permission is hereby granted|all rights reserved|"
    r"redistribution and use in source",
    re.IGNORECASE,
)


def _strip_leading_license_comment(text: str, *, max_scan_lines: int = 60) -> str:
    """Strip a leading license/copyright comment block, if one is present.

    Real code corpora (github-code especially) share huge amounts of
    near-identical license-header boilerplate across unrelated files -- the
    standard Apache-2.0/MIT header text is byte-for-byte identical across
    thousands of real repositories (verified directly: a real 100-file
    sample of this project's own curated allowlist found 43 files with a
    leading license/copyright block, two of which were the exact same
    Apache Software Foundation boilerplate). That inflates near-duplicate
    rejection for content that has nothing to do with the actual code, and
    spends real training tokens on repeated legal text this project has no
    use for -- license provenance is already tracked separately, in each
    `Document`'s own `license` field, not learned from the file text.

    Conservative by design: only strips a single contiguous comment block
    (a C-style block comment, an HTML comment, a triple-quoted docstring,
    or a run of hash/double-slash lines) at the very start of the file, and
    only when that block actually contains a real license/copyright signal
    word -- a normal leading module docstring or file-purpose comment is
    left untouched. Never scans past `max_scan_lines`, so a pathological
    all-comment file cannot make this silently consume the whole thing.
    """

    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines) or index >= max_scan_lines:
        return text
    first = lines[index].lstrip()
    scan_limit = min(len(lines), index + max_scan_lines)

    if first.startswith("/*"):
        block_end = None
        for offset in range(index, scan_limit):
            if "*/" in lines[offset]:
                block_end = offset + 1
                break
        if block_end is None:
            return text
    elif first.startswith("<!--"):
        block_end = None
        for offset in range(index, scan_limit):
            if "-->" in lines[offset]:
                block_end = offset + 1
                break
        if block_end is None:
            return text
    elif first.startswith('"""') or first.startswith("'''"):
        quote = first[:3]
        if first.count(quote) >= 2:
            block_end = index + 1
        else:
            block_end = None
            for offset in range(index + 1, scan_limit):
                if quote in lines[offset]:
                    block_end = offset + 1
                    break
            if block_end is None:
                return text
    elif first.startswith("#") or first.startswith("//"):
        marker = "#" if first.startswith("#") else "//"
        offset = index
        while offset < scan_limit and (
            not lines[offset].strip() or lines[offset].lstrip().startswith(marker)
        ):
            offset += 1
        block_end = offset
    else:
        return text

    block_text = "".join(lines[index:block_end])
    if not _LICENSE_SIGNAL_PATTERN.search(block_text):
        return text
    remainder_start = block_end
    while remainder_start < len(lines) and not lines[remainder_start].strip():
        remainder_start += 1
    return "".join(lines[remainder_start:])


def iter_github_code(
    *,
    languages: Iterable[str] | None = None,
    repo_names: Iterable[str] | None = None,
    limit: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10_000,
) -> Iterator[Document]:
    """Stream github-code, admitting only permissively-licensed rows.

    Every row is filtered client-side (the dataset's own `languages=`/
    `licenses=` `load_dataset` kwargs were verified empirically to NOT
    filter under `streaming=True` on this pinned revision -- see
    `_GITHUB_CODE_PERMISSIVE_LICENSES`'s comment). A row whose real license
    does not normalize to one of this project's `PERMISSIVE_CODE_LICENSES`
    is skipped before `Document.create` ever sees it -- letting the
    exception path do this filtering would crash the whole stream on the
    first non-permissive row instead of just skipping it. `repo_name` is
    checked against `_MANUALLY_VERIFIED_REPO_LICENSES` first, overriding the
    dataset's own (possibly stale or undetected) `license` field for the
    small, explicit set of repos independently verified there -- see that
    dict's own comment for why and how each was checked.

    `languages`, when given, restricts to those languages (matched
    case-insensitively against the dataset's own `language` field, e.g.
    `{"Python", "JavaScript"}`). `repo_names`, when given, restricts to
    exactly those `owner/repo` strings -- the mechanism for pulling only a
    curated allowlist of well-known repositories out of this otherwise huge
    dataset, rather than an unfiltered crawl. Like `iter_dclm_edu`,
    `start`/`limit` count only admitted (post-filter) rows, not raw stream
    position -- appropriate here even more than for DCLM-Edu, since far more
    of the raw stream gets rejected (wrong license, wrong language, or not
    in the curated allowlist) than admitted.
    """

    from datasets import load_dataset

    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if start < 0:
        raise ValueError("start cannot be negative")
    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be positive")
    language_filter = {name.lower() for name in languages} if languages is not None else None
    repo_filter = set(repo_names) if repo_names is not None else None
    dataset = load_dataset(
        GITHUB_CODE_DATASET,
        revision=GITHUB_CODE_REVISION,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    emitted = 0
    admitted_index = 0
    for row in dataset:
        repo_name = str(row["repo_name"])
        normalized_license = _MANUALLY_VERIFIED_REPO_LICENSES.get(repo_name)
        if normalized_license is None:
            license_key = str(row["license"]).lower()
            normalized_license = _GITHUB_CODE_PERMISSIVE_LICENSES.get(license_key)
        if normalized_license is None:
            continue
        if language_filter is not None and str(row["language"]).lower() not in language_filter:
            continue
        if repo_filter is not None and repo_name not in repo_filter:
            continue
        if admitted_index < start:
            admitted_index += 1
            continue
        if limit is not None and emitted >= limit:
            return
        yield Document.create(
            _strip_leading_license_comment(str(row["code"])),
            source=f"https://github.com/{repo_name}",
            revision=GITHUB_CODE_REVISION,
            license=normalized_license,
            language=str(row["language"]),
            record_id=f"{repo_name}:{row['path']}",
            path=str(row["path"]),
            source_type="code",
        )
        admitted_index += 1
        emitted += 1


def iter_ebook_markdown(
    directory: str | Path,
    *,
    license: str = "Public Domain",
    revision: str = "n/a",
    language: str = "English",
    limit: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
) -> Iterator[Document]:
    """Read already-produced ``book.md`` files from a local ebook-ingestion output tree.

    This is a filesystem adapter, not a network source: it reads the plain
    Markdown files an already-completed run of the standalone
    ``pdf-to-markdown-rag`` pipeline (``docs/pdf-to-markdown-rag.zip``) wrote to
    ``directory``, at ``directory/md/<book-id>/book.md`` per book. It never
    imports that pipeline's own code and never touches its retrieval-oriented
    outputs (``chunks/*/chunks.jsonl``, the optional SQLite FTS5/vector corpus)
    -- those serve RAG retrieval at inference time, a different consumption
    pattern than pretraining, which just wants each book as one continuous
    document (MF-124, 2026-09-12 decision).

    The pipeline's own per-book ``metadata/<book-id>/license.json`` sidecar is
    deliberately not read either: MF-124 restricts real ingestion to books
    whose public-domain status is asserted by the person curating the input
    ``directory`` before this function ever runs, not detected from a file --
    the same explicit assertion this project already leans on for the
    ``_MANUALLY_VERIFIED_REPO_LICENSES`` override table. ``license``,
    ``revision``, and ``language`` therefore apply uniformly to every book
    this call yields; run it once per language/rights-basis batch if a real
    corpus needs to mix them.

    ``revision`` has no natural meaning for an ebook (no repository commit or
    dataset snapshot to pin) -- the default ``"n/a"`` is a placeholder that
    only needs to be non-empty to satisfy ``Document.__post_init__``.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if start < 0:
        raise ValueError("start cannot be negative")
    book_paths = sorted(Path(directory).glob("md/*/book.md"))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(book_paths)
    emitted = 0
    for index, book_path in enumerate(book_paths):
        if index < start:
            continue
        if limit is not None and emitted >= limit:
            return
        text = book_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        book_id = book_path.parent.name
        yield Document.create(
            text,
            source=f"ebook:{book_id}",
            revision=revision,
            license=license,
            language=language,
            record_id=book_id,
            path=str(book_path),
            source_type="text",
        )
        emitted += 1


def filter_and_deduplicate(
    documents: Iterable[Document],
    *,
    min_characters: int = 32,
    max_characters: int = 1_000_000,
    excluded_hashes: frozenset[str] | set[str] = frozenset(),
) -> Iterator[Document]:
    """Drop junk and exact duplicates, keeping the first copy of anything repeated.

    ``excluded_hashes`` is the contamination guard: pass the validation or
    benchmark hashes in here and those documents can never leak into training,
    which would otherwise make the evaluation scores meaningless.
    """

    if min_characters < 0 or max_characters < min_characters:
        raise ValueError("invalid character bounds")
    # Hashes seen so far in this stream. Exact-duplicate removal only; near-
    # duplicate detection is a bigger job and lives in `shards.py`.
    seen: set[str] = set()
    for document in documents:
        length = len(document.text)
        if length < min_characters or length > max_characters:
            continue
        if not document.text.strip() or "\x00" in document.text:
            continue
        if document.content_hash in seen:
            continue
        if document.content_hash in excluded_hashes:
            continue
        seen.add(document.content_hash)
        yield document


def split_documents(
    documents: Iterable[Document],
    *,
    validation_fraction: float = 0.01,
) -> tuple[list[Document], list[Document]]:
    """Assign each document to train or validation by hashing its content.

    Hashing instead of shuffling has a property that matters: the same document
    always lands on the same side, no matter what order it arrives in, how many
    other documents there are, or how many times the pipeline is re-run. That is
    what keeps a validation set honest across re-runs.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    train: list[Document] = []
    validation: list[Document] = []
    threshold = int(validation_fraction * 10_000)
    for document in documents:
        bucket = split_bucket(document)
        if bucket < threshold:
            validation.append(replace(document, split="validation"))
        else:
            train.append(replace(document, split="train"))
    return train, validation


def split_bucket(document: Document) -> int:
    """Return a stable split bucket that survives provenance-preserving transforms.

    Bucket 0-9,999 derived from the first 8 hex characters of the hash. Using the
    *parent* hash when one exists is the important part: a FIM-rewritten copy of a
    training document must not be able to land in validation, or the model would
    be graded on text it has effectively already read.
    """

    identity_hash = document.parent_content_hash or document.content_hash
    return int(identity_hash[:8], 16) % 10_000


def pack_documents(
    documents: Iterable[Document],
    tokenizer: MiniFrontierTokenizer,
    *,
    sequence_length: int,
    drop_remainder: bool = True,
) -> Iterator[PackedSequence]:
    """Glue tokenized documents into one ribbon and slice fixed-length sequences.

    Padding every document to ``sequence_length`` would spend most of the training
    compute on padding tokens. Packing spends none: every position in every
    sequence is a real token the model can learn from. The cost is that documents
    get split across sequence boundaries, and that a sequence can contain the tail
    of one document and the head of another -- with ``<|eos|>`` between them, so
    the model can at least learn where the seam is.
    """

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    buffer: list[int] = []
    for document in documents:
        # add_eos marks the end of this document inside the continuous ribbon.
        buffer.extend(tokenizer.encode(document.text, add_eos=True))
        # Emit as many complete sequences as the buffer can supply, then keep the
        # leftover to be continued by the next document.
        while len(buffer) >= sequence_length:
            yield PackedSequence(tuple(buffer[:sequence_length]), sequence_length)
            del buffer[:sequence_length]
    # The final scrap is normally dropped: it is a rounding error's worth of
    # tokens, and padding it would introduce the only padded batch in the run.
    if buffer and not drop_remainder:
        non_padding = len(buffer)
        buffer.extend([tokenizer.pad_id] * (sequence_length - len(buffer)))
        yield PackedSequence(tuple(buffer), non_padding)


class PackedTokenDataset(IterableDataset[torch.Tensor]):
    """Worker-sharded iterable over already deterministic packed sequences.

    With several DataLoader worker processes, each one walks the same stream and
    keeps only every Nth sequence. Simple, and it guarantees no sequence is
    delivered twice -- which would quietly train on the same tokens more than once.
    """

    def __init__(self, sequences: Iterable[PackedSequence]) -> None:
        super().__init__()
        self._sequences = sequences

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        for index, sequence in enumerate(self._sequences):
            if worker is None or index % worker.num_workers == worker.id:
                yield sequence.tensor()
