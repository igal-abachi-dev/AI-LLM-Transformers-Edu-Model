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
import shutil
import sys
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import IterableDataset, get_worker_info

from minifrontier import git_utils
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


def iter_parquet_documents(path: str | Path) -> Iterator[Document]:
    """Read real `Document` rows back from a Parquet file written by
    `shards.ParquetDocumentWriter` (MF-125) -- the read-side counterpart,
    used by `iter_github_code_from_repos`'s own document cache (MF-134;
    see `docs/IMPLEMENTATION_DECISIONS.md`'s "Data storage formats" note for
    why Parquet, not JSONL, is the right format for this local-reuse cache).
    Also the real, local half of what MF-132 scoped as a future parquet
    re-import loader for a *published* HF dataset repo -- this covers the
    local-file case directly; a remote `datasets.load_dataset(repo_id, ...)`
    path remains MF-132's own separate, not-yet-built scope.
    """

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    for row in table.to_pylist():
        yield Document.from_mapping(row)


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


# MF-070 (2026-09-15): `codeparrot/github-code`'s own HF dataset card was
# fetched directly and confirmed to be a one-time static BigQuery snapshot
# (v1.1 queried 2022-03-16, no stated refresh mechanism) -- a wrong source
# for a curated repo allowlist, not merely a slow one: even a successful
# `iter_github_code` scan returns multi-year-old code, and finding 161
# specific repos inside its full ~115M-file crawl is what made it real-
# measured 10-30x slower than every other MF-070 mixture source. This block
# replaces that path for the allowlist case with a direct, real, current
# `git clone --depth 1` per repo -- no GitHub API, no credentials, no rate
# limit: cloning is a different subsystem from `api.github.com`'s 60/hr
# unauthenticated REST cap, and per-repo license detection reads the real
# `LICENSE`-family file already sitting in the clone instead of calling the
# API at all (the same real method this file's own `_MANUALLY_VERIFIED_
# REPO_LICENSES` entries were each individually verified by -- automated
# here for all 161 repos rather than kept as a 13-repo manual list).
_GITHUB_EXTENSION_LANGUAGES: Final[dict[str, str]] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".cs": "C#",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".hxx": "C++",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    # MF-121's real allowlist additions (2026-09-12: C, SQL, Markdown,
    # Dockerfile, CMake, PowerShell, Shell, Batchfile, TeX) were missing
    # here entirely until this real, found-by-inspection gap (2026-09-15,
    # a direct question -- "do we miss any extension" -- prompted actually
    # re-checking the allowlist file rather than trusting the original
    # 12-language framing). Without these, libuv/cJSON/openssl (C),
    # supabase/sqlfluff (SQL), the two Markdown/CMake/PowerShell/Shell/
    # Batchfile/TeX repos would each clone successfully but yield zero
    # documents -- every one of their real files silently unmatched.
    ".c": "C",
    # A real, inherent ambiguity, not fully resolvable from the extension
    # alone: plain `.h` headers are used by both C and C++ in practice.
    # Mapped to C here since the allowlist's own C++ entries mostly use
    # `.hpp`/`.hh`/`.hxx` for headers specifically (already covered above),
    # while the three real C repos (libuv, cJSON, openssl) use plain `.h`
    # as their primary header convention -- a defensible, disclosed choice,
    # not a claim of perfect accuracy.
    ".h": "C",
    ".sql": "SQL",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".cmake": "CMake",
    ".ps1": "PowerShell",
    ".psm1": "PowerShell",
    ".psd1": "PowerShell",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".bat": "Batchfile",
    ".cmd": "Batchfile",
    ".tex": "TeX",
}
# Filename-based matches, for the two real MF-121 categories with no
# distinctive extension of their own: a literal `Dockerfile` (or a real,
# common variant like `Dockerfile.dev`/`Dockerfile.alpine`) and
# `CMakeLists.txt`, CMake's own standard, exact build-file name. Checked
# against `file_path.name`, not `.suffix`, after the extension dict misses.
_GITHUB_DOCKERFILE_NAME_PREFIX: Final = "dockerfile"
_GITHUB_CMAKELISTS_NAME: Final = "cmakelists.txt"
# Real, curated per this project's own real allowlist (`configs/
# code-repo-allowlist.txt`) -- no generic catch-all extension list, since an
# unrecognized extension is meant to be skipped, not mislabeled.
# Extended 2026-09-15 (MF-070) from a real `git ls-tree -r -l HEAD` byte-size
# scan across all 166 then-cached mirrors, not guessed: `docs`/`doc`/
# `documentation` (1.1GB+170MB+75MB across 95/25/13 repos, prose, not code),
# `testdata`/`test-data`/`fixtures`/`testfixtures`/`__snapshots__`/
# `snapshots` (real binary/data fixtures -- confirmed concretely by the two
# LFS incidents found the same day, `qdrant/qdrant`'s
# `tests/e2e_tests/test_data/storage.tar.xz` and `microsoft/vscode`'s
# `extensions/copilot/test/simulation/cache`), and `assets`/`images`/`img`/
# `media` (binary, not text/code). `test`/`tests`/`__tests__` themselves
# deliberately NOT added, after reconsidering: real source files inside them
# (assertions, real API usage) are genuine, in-scope training signal, same
# reasoning already applied to `examples`/`samples` (177MB/119MB, 59/20
# repos, also not added) -- there is no reliable general way to separate
# "real usage" test code from "dummy" test scaffolding by directory/file
# structure alone (would need per-language semantic analysis: different
# mocking-library conventions per language, fragile and unexplainable), so
# the fallback is to keep all real source files under `test`/`tests` and
# only strip the non-source content in and around them (fixtures, binary
# data, docs, images) covered by the other names in this set.
_GITHUB_CLONE_SKIP_DIRS: Final = frozenset(
    {
        ".git",
        "node_modules",
        "vendor",
        "dist",
        "build",
        "target",
        "bin",
        "obj",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "coverage",
        "testdata",
        "test-data",
        "fixtures",
        "testfixtures",
        "__snapshots__",
        "snapshots",
        "docs",
        "doc",
        "documentation",
        "assets",
        "images",
        "img",
        "media",
    }
)
_GITHUB_MAX_FILE_BYTES: Final = 1_000_000
# Defensive cap against generated/minified/vendored data files a shallow
# clone can still contain despite the directory skip-list above.
_GITHUB_MAX_LINE_CHARS: Final = 1000
# Mirrors codeparrot/github-code's own real, disclosed filtering convention
# (its dataset card states it drops files with lines over 1000 characters),
# kept here for parity now that this project builds its own equivalent.
_GITHUB_LICENSE_FILENAME_PREFIXES: Final = ("license", "licence", "copying")
# Case-insensitive prefix match, not an exact enumerated filename list --
# real variance already found in this project's own manual-verification
# research (`_MANUALLY_VERIFIED_REPO_LICENSES`'s comments): `LICENSE`,
# `LICENSE.md`, `LICENSE.txt`, `Licence.txt` (British spelling), `COPYING`.

_LICENSE_TEXT_SIGNATURES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    # Checked in order; the first match wins. More specific/distinctive
    # signatures are listed before more generic ones that could otherwise
    # false-positive on a shared opening clause (BSD-3 before BSD-2, in
    # particular, since BSD-2's text is a strict prefix of BSD-3's).
    ("Apache-2.0", ("apache license", "version 2.0")),
    (
        "BSD-3-Clause",
        (
            "redistribution and use in source and binary forms",
            "neither the name of",
        ),
    ),
    (
        "BSD-2-Clause",
        ("redistribution and use in source and binary forms",),
    ),
    ("Unlicense", ("this is free and unencumbered software released into the public domain",)),
    ("CC0-1.0", ("cc0 1.0 universal",)),
    (
        "ISC",
        (
            "permission to use, copy, modify, and/or distribute this software",
            "with or without fee is hereby granted",
        ),
    ),
    (
        "MIT",
        ("permission is hereby granted, free of charge, to any person obtaining a copy",),
    ),
)


def _detect_license_from_text(text: str) -> str | None:
    """Classify real license file text via its own canonical, distinctive wording.

    Conservative by design: returns `None` (never a guess) when no real
    signature phrase is found, exactly like `iter_github_code`'s own
    dataset-field gate above -- a repo whose license can't be confidently
    classified this way is skipped, not silently admitted.
    """

    lowered = " ".join(text.lower().split())
    for license_name, signatures in _LICENSE_TEXT_SIGNATURES:
        if all(signature in lowered for signature in signatures):
            return license_name
    return None


_NON_PERMISSIVE_LICENSE_SIGNATURES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("GPL-3.0", ("gnu general public license", "version 3")),
    ("GPL-2.0", ("gnu general public license", "version 2")),
    ("AGPL-3.0", ("gnu affero general public license",)),
    ("LGPL", ("gnu lesser general public license",)),
    ("MPL-2.0", ("mozilla public license", "version 2.0")),
)


def _resolve_repo_license(repo_name: str, repo_root: Path) -> str | None:
    """Real per-repo license, trusting `configs/code-repo-allowlist.txt`'s own
    curation rather than re-litigating it per repo.

    `configs/code-repo-allowlist.txt` was already compiled with a real
    permissive-license check per repo (see its own header comment) -- this
    is not a second, independent gate on top of that. The manual-override
    table (human-verified for 13 repos where automated detection failed) is
    checked first; otherwise a best-effort real read of the cloned repo's
    own LICENSE-family file fills in the specific label when its wording is
    clean and canonical (true for the large majority of these well-known
    projects). When that text can't be confidently classified as one of
    `PERMISSIVE_CODE_LICENSES` either way, the repo is still admitted here
    (trusting the allowlist, quietly) -- the one thing actively checked and
    loudly logged is real drift since curation: current LICENSE text that
    now reads as a copyleft license. Never a network call.
    """

    manual = _MANUALLY_VERIFIED_REPO_LICENSES.get(repo_name)
    if manual is not None:
        return manual
    try:
        entries = list(repo_root.iterdir())
    except OSError:
        entries = []
    license_text = None
    for entry in entries:
        if entry.is_file() and entry.name.lower().startswith(_GITHUB_LICENSE_FILENAME_PREFIXES):
            try:
                license_text = entry.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            break
    if license_text is not None:
        detected = _detect_license_from_text(license_text)
        if detected is not None:
            return detected
        lowered = " ".join(license_text.lower().split())
        for bad_license, signatures in _NON_PERMISSIVE_LICENSE_SIGNATURES:
            if all(signature in lowered for signature in signatures):
                print(
                    f"WARNING: {repo_name}'s real LICENSE file now reads as {bad_license}, "
                    "not permissive -- it was allowlisted as permissive; excluding it from "
                    "this run. It may have relicensed since configs/code-repo-allowlist.txt "
                    "was curated and is worth a real, manual re-check.",
                    file=sys.stderr,
                )
                return None
    # Curated allowlist entry, but the real LICENSE text (if any was found)
    # doesn't match this project's own canonical-wording patterns either
    # way -- the same real false-negative pattern _MANUALLY_VERIFIED_REPO_
    # LICENSES' 13 entries already document (non-canonical phrasing,
    # unusual file location). Deliberately NOT defaulted to a guessed
    # specific label (would put a possibly-wrong SPDX id into permanent
    # provenance/a published dataset card -- worse than omitting this repo):
    # skipped this run, same as a clone failure, not fabricated. A repo that
    # lands here repeatedly is a real, cheap candidate for a one-line
    # addition to `_MANUALLY_VERIFIED_REPO_LICENSES` (read its real LICENSE
    # file once, by hand, exactly like the existing 13).
    return None


def _extract_repo_documents(names: list[str], *, clone_repo, resolve_license) -> Iterator[Document]:
    """Real per-repo extraction, unfiltered by `languages`/`start`/`limit` --
    every admitted file from every repo, in `names`' own order. Kept
    separate from `iter_github_code_from_repos` so its own document cache
    (MF-134) can wrap this raw stream once and be replayed with a different
    `languages`/`start`/`limit` later without touching git at all.
    """

    for repo_name in names:
        with tempfile.TemporaryDirectory(prefix="minifrontier-github-clone-") as tmp_dir:
            destination = Path(tmp_dir) / "repo"
            try:
                yield from _extract_one_repo_documents(
                    repo_name, destination, clone_repo=clone_repo, resolve_license=resolve_license
                )
            except Exception as error:
                # The whole per-repo body (clone, license resolution, and
                # file walking) is covered by one outer catch, not just the
                # clone step -- a real, previously-unprotected gap: an
                # unexpected exception anywhere past the clone (a
                # `resolve_license` edge case, a file-walking surprise) used
                # to propagate all the way up and crash this whole
                # generator, aborting every remaining repo in a real,
                # multi-hour, 161-repo run over one bad one. `--quiet` on
                # the real git commands only suppresses progress/status
                # noise -- real errors always reach stderr regardless, and
                # `capture_output=True` does capture them; this is the real
                # place that was silently discarding them. `stderr` is only
                # present on `subprocess.CalledProcessError`-family
                # exceptions; `getattr` degrades safely for anything else
                # (e.g. `FileNotFoundError` if git itself isn't installed).
                stderr = getattr(error, "stderr", None)
                detail = stderr.decode("utf-8", errors="replace").strip() if stderr else str(error)
                print(f"WARNING: skipping {repo_name}: {detail}", file=sys.stderr)


def _extract_one_repo_documents(
    repo_name: str, destination: Path, *, clone_repo, resolve_license
) -> Iterator[Document]:
    """One repo's real extraction -- clone, resolve its license, walk its
    admitted files. Split out of `_extract_repo_documents`'s own loop so
    that function's outer `except Exception` genuinely covers every step
    for this one repo, not just the clone call.
    """

    commit_sha = clone_repo(repo_name, destination)
    license_value = resolve_license(repo_name, destination)
    if license_value is None:
        return
    clone_root = destination.resolve()
    for file_path in sorted(destination.rglob("*")):
        # Never follow a symlink: git checks symlinks out as real filesystem
        # links on Linux/WSL/macOS (Windows checks them out as small text files
        # by default, so this is a real, platform-dependent gap, not a
        # hypothetical one), so a repo file `x.py -> ../../../../etc/passwd` (or
        # any absolute/out-of-tree target) would otherwise be read from OUTSIDE
        # the clone and written into the training corpus. `Path.is_file()`
        # follows symlinks, so this check must come before it, not rely on it.
        if file_path.is_symlink():
            continue
        if not file_path.is_file():
            continue
        if not file_path.resolve().is_relative_to(clone_root):
            continue
        relative_path = file_path.relative_to(destination)
        # Case-insensitive: real, confirmed case variants exist across these
        # 166 repos for these exact names (e.g. `Tests` in 14 repos, `Assets`
        # in 13, `TestData` in 9 -- a 2026-09-15 real scan, not assumed), so a
        # case-sensitive match against the all-lowercase skip set above would
        # silently miss them.
        if any(part.lower() in _GITHUB_CLONE_SKIP_DIRS for part in relative_path.parts):
            continue
        language = _GITHUB_EXTENSION_LANGUAGES.get(file_path.suffix.lower())
        if language is None:
            lowered_name = file_path.name.lower()
            if lowered_name.startswith(_GITHUB_DOCKERFILE_NAME_PREFIX):
                language = "Dockerfile"
            elif lowered_name == _GITHUB_CMAKELISTS_NAME:
                language = "CMake"
        if language is None:
            continue
        # One bad file must not lose the rest of an otherwise-good repo,
        # the same real principle already applied one level up (one bad
        # repo must not lose the rest of a 161-repo run). The narrower
        # (UnicodeDecodeError, OSError) catch below already covered the
        # read step specifically; this wider one covers everything after
        # it too (line-length scan, license-header stripping, `Document.
        # create`'s own validation) -- silent on failure, matching this
        # same per-file loop's own already-established quiet-skip
        # convention (unlike the louder, repo-level diagnostic, a single
        # bad file among possibly thousands is not worth a warning print).
        try:
            document = _document_for_file(
                file_path,
                relative_path,
                repo_name=repo_name,
                commit_sha=commit_sha,
                license_value=license_value,
                language=language,
            )
        except Exception:
            continue
        if document is not None:
            yield document


def _document_for_file(
    file_path: Path,
    relative_path: Path,
    *,
    repo_name: str,
    commit_sha: str,
    license_value: str,
    language: str,
) -> Document | None:
    try:
        if file_path.stat().st_size > _GITHUB_MAX_FILE_BYTES:
            return None
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    if any(len(line) > _GITHUB_MAX_LINE_CHARS for line in text.splitlines()):
        return None
    text = _strip_leading_license_comment(text)
    if not text.strip():
        return None
    return Document.create(
        text,
        source=f"https://github.com/{repo_name}",
        revision=commit_sha,
        license=license_value,
        language=language,
        record_id=f"{repo_name}:{relative_path.as_posix()}",
        path=relative_path.as_posix(),
        source_type="code",
    )


def _document_cache_parts_dir(document_cache_path: Path) -> Path:
    return document_cache_path.with_name(f"{document_cache_path.stem}.parts")


def _sanitize_repo_name_for_parquet_part(repo_name: str) -> str:
    return repo_name.replace("/", "__") + ".parquet"


def _extract_repo_documents_resumable(
    names: list[str],
    document_cache_path: Path,
    *,
    clone_repo,
    resolve_license,
    force_refresh: bool,
) -> Iterator[Document]:
    """Real, per-repo-checkpointed replacement for
    `_extract_repo_documents` + `_tee_to_parquet_document_cache`'s combined,
    whole-run-spanning write (2026-09-15, MF-134 follow-up, user-requested
    after a real, second multi-hour interruption lost all progress: the
    document cache's own staging `.tmp` file is written by one
    `ParquetDocumentWriter` spanning the *entire* run and only gets a valid
    footer once `finalize()` runs at the very end -- a hard process kill
    (not a clean Python exception; verified directly, real: confirmed via
    `pyarrow.parquet.ParquetFile` raising `ArrowInvalid: Parquet magic
    bytes not found in footer` on a genuine `exit -1` kill's leftover
    `.tmp`) leaves that file permanently unreadable and unresumable,
    regardless of how much real work it represents).

    One real, independently-finalized Parquet file per repo under
    `document_cache_path`'s own sibling `<stem>.parts/` directory, named
    deterministically from the repo name -- the *existence* of a repo's own
    part file, under its own final (non-`.tmp`) name, is itself the
    "already done" signal; no separate manifest file is needed or kept, so
    there is nothing that could drift out of sync with the real files on
    disk. Each part is written via `ParquetDocumentWriter`'s own existing
    atomic staging+rename (`finalize()`), the same mechanism already used
    for the final combined cache -- a hard kill mid-write of any *one*
    repo's part leaves only that repo's own orphaned `.tmp` (cleaned up by
    a later attempt at that same repo, the same self-healing principle
    already used for the git-mirror cache's own staging path) and never
    touches any other, already-completed repo's part. Worst case lost work
    on a crash: one repo's extraction, not the whole run.

    A repo that fails entirely (clone error, unresolved license) still
    gets a real, valid, empty-schema part file written for it (
    `ParquetDocumentWriter.finalize()`'s own existing empty-case handling)
    -- matching this project's already-established "a failed repo is
    skipped for good, not retried forever" contract (the existing
    non-resumable design already had this property implicitly, since a
    fully-replayed cache never re-attempts a repo that yielded zero
    documents the first time either; this preserves it exactly, not a new
    behavior).

    `force_refresh=True` discards any existing `parts_dir` outright and
    starts every repo over, matching `iter_github_code_from_repos`'s own
    existing "bypasses both caches unconditionally" contract.

    The final, single combined `document_cache_path` file (the one
    `iter_github_code_from_repos`'s fast-path replay actually checks for)
    is only ever built once every repo in `names` has a real part on this
    call -- via `_finalize_document_cache_from_parts` below -- preserving
    the existing, already-established contract that the shared cache only
    reflects a *fully* completed run, never a partial one (an early stop
    via `limit`, or a hard kill mid-run, both correctly leave no complete
    `document_cache_path` behind, exactly as today).
    """

    from minifrontier.shards import ParquetDocumentWriter

    parts_dir = _document_cache_parts_dir(document_cache_path)
    if force_refresh and parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    for repo_name in names:
        part_path = parts_dir / _sanitize_repo_name_for_parquet_part(repo_name)
        if part_path.exists() and not force_refresh:
            yield from iter_parquet_documents(part_path)
            continue
        # Self-heal a leftover `.tmp` from an interrupted part-write for
        # this exact repo (a real, if narrow, race -- the same principle
        # already applied to the git-mirror cache's own staging path).
        part_path.with_name(f".{part_path.name}.tmp").unlink(missing_ok=True)
        repo_documents = list(
            _extract_repo_documents(
                [repo_name], clone_repo=clone_repo, resolve_license=resolve_license
            )
        )
        part_writer = ParquetDocumentWriter(part_path)
        for document in repo_documents:
            part_writer.add(document)
        part_writer.finalize()
        yield from repo_documents

    _finalize_document_cache_from_parts(document_cache_path, parts_dir, names)


def _finalize_document_cache_from_parts(
    document_cache_path: Path, parts_dir: Path, names: list[str]
) -> None:
    """Concatenate every real, already-finalized per-repo part into the
    single final cache file, one part's documents at a time (not every
    part loaded into memory simultaneously) -- reusing
    `ParquetDocumentWriter`'s own existing atomic staging+rename, so an
    interruption during this final merge itself leaves no partial
    `document_cache_path` behind either, exactly like every other atomic
    publish in this project. `parts_dir` cleanup afterward is best-effort
    and deliberately non-load-bearing: `document_cache_path`'s own
    existence is the sole authority `iter_github_code_from_repos` already
    checks for "complete," so a failed cleanup here (a locked file,
    whatever) cannot make a genuinely complete cache look incomplete.
    """

    from minifrontier.shards import ParquetDocumentWriter

    writer = ParquetDocumentWriter(document_cache_path)
    for repo_name in names:
        part_path = parts_dir / _sanitize_repo_name_for_parquet_part(repo_name)
        for document in iter_parquet_documents(part_path):
            writer.add(document)
    writer.finalize()
    shutil.rmtree(parts_dir, ignore_errors=True)


_GITHUB_CACHE_DEFAULT_DOCUMENT_CACHE: Final = Path("data/github-code-cache/documents.parquet")


def iter_github_code_from_repos(
    repo_names: Iterable[str],
    *,
    languages: Iterable[str] | None = None,
    limit: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
    cache_dir: Path | None = git_utils.GITHUB_CACHE_DEFAULT_DIR,
    document_cache_path: Path | None = _GITHUB_CACHE_DEFAULT_DOCUMENT_CACHE,
    force_refresh: bool = False,
    max_staleness_seconds: float = git_utils.GITHUB_CACHE_DEFAULT_MAX_STALENESS_SECONDS,
    clone_repo=None,
    resolve_license=_resolve_repo_license,
) -> Iterator[Document]:
    """Stream real, current source files by cloning each repo directly.

    Unlike `iter_github_code` (a static 2022-03-16 snapshot, see the module
    note above), every file yielded here comes from a real, current clone of
    its own repository, with a real per-repo commit SHA as `revision` -- a
    precise, individual provenance record, unlike `iter_github_code`'s
    single shared dataset-repo revision stamped on every file regardless of
    which real GitHub commit it actually came from.

    Two independent caching layers (MF-134), both real and both on by
    default:
    - `cache_dir` (Layer 0): a persistent, per-repo bare mirror clone
      (`git clone --mirror` once, `git fetch` on later calls once the cache
      has gone stale past `max_staleness_seconds`) -- bounds how often this
      pipeline contacts GitHub at all, not just how long one run takes.
    - `document_cache_path` (Layer 1): a real Parquet dump (zstd-compressed,
      via `shards.ParquetDocumentWriter`) of every already-extracted
      document (post-clone, post-filter, pre-tokenization) -- a later call
      with a real, existing cache file at this path replays it directly (no
      git, no repo walking, no network at all) rather than re-extracting
      from the repos again. Parquet, not JSONL, matches real established
      practice for exactly this kind of local reuse cache (verified: the
      HuggingFace `datasets` library's own local cache is Arrow/Parquet-
      family for the same reason -- fast, columnar, memory-mappable reads;
      JSONL.zst is the right choice for a *published, static* raw-corpus
      dump, e.g. Pile/RedPajama/Dolma, a different real use case from this
      one). `languages`/`start`/`limit` are still applied fresh on top of a
      cache replay, so the same cache serves a differently-filtered rebuild
      without invalidation. Set to `None` to disable (always re-extract from
      the real repos).

    `force_refresh=True` bypasses both caches unconditionally: every mirror
    is re-cloned (Layer 0) and the document cache is rebuilt from that fresh
    extraction (Layer 1) rather than replayed.

    `repo_names` (required, unlike `iter_github_code`'s optional allowlist --
    a direct clone has nothing to iterate without a specific target list)
    are cloned in order, or shuffled first when `shuffle_seed` is given (a
    cache replay reuses whichever order originally produced the cache,
    since it was already fixed at write time -- pass `force_refresh=True`
    for a genuinely different order). `languages`/`start`/`limit` behave
    exactly like `iter_github_code`'s own (case-insensitive language match;
    `start`/`limit` count only admitted, post-filter documents).
    `clone_repo`/`resolve_license` are injectable purely for testing -- real
    callers never need to pass them.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if start < 0:
        raise ValueError("start cannot be negative")
    names = list(repo_names)
    if not names:
        raise ValueError("repo_names must be non-empty")
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(names)
    language_filter = {name.lower() for name in languages} if languages is not None else None
    if clone_repo is None:

        def clone_repo(repo_name: str, destination: Path) -> str:
            return git_utils.clone_via_cached_mirror(
                repo_name,
                destination,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
                max_staleness_seconds=max_staleness_seconds,
            )

    use_cache = (
        document_cache_path is not None and not force_refresh and document_cache_path.exists()
    )
    if use_cache:
        documents = iter_parquet_documents(document_cache_path)
    elif document_cache_path is not None:
        documents = _extract_repo_documents_resumable(
            names,
            document_cache_path,
            clone_repo=clone_repo,
            resolve_license=resolve_license,
            force_refresh=force_refresh,
        )
    else:
        documents = _extract_repo_documents(
            names, clone_repo=clone_repo, resolve_license=resolve_license
        )

    admitted_index = 0
    emitted = 0
    for document in documents:
        if limit is not None and emitted >= limit:
            return
        if language_filter is not None and document.language.lower() not in language_filter:
            continue
        if admitted_index < start:
            admitted_index += 1
            continue
        yield document
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
