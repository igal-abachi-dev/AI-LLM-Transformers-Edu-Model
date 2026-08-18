# Data governance

MiniFrontier trains only on sources whose use can be explained and reproduced. A dataset being publicly downloadable is not sufficient authorization to train on it.

## Required source manifest

Every source must record:

- stable source name and URL or local collection identifier;
- immutable revision, commit, snapshot, or dataset version;
- license identifier and a link or copy of the governing terms;
- language and source type;
- original path or record identifier;
- content hash after canonical byte encoding;
- acquisition date and the pipeline version that admitted it;
- train, validation, or evaluation purpose.

Records missing source, revision, license, or content hash are rejected. Code sources are limited to an approved allow-list of permissive or public-domain licenses. The Stack and similar mixed-license aggregations are not default V1 sources.

## Processing rules

1. Validate the immutable source revision, license, and required provenance.
2. Apply basic structural checks for size, encoding, empty/malformed content, and exact exclusions.
3. Apply source-specific admission filters for secrets/credentials, the approved personal-data
   policy, generated spam, and other source-specific risks.
4. Deduplicate before assigning splits.
5. Isolate evaluation fixtures and their exact/near-duplicates from training.
6. Tokenize and write immutable shards only after the split is frozen.
7. Retain aggregate filtering counts, pipeline versions, manifests, and hashes—not rejected
   sensitive content.

## Implementation status and task ownership

The current `filter_and_deduplicate()` implementation is intentionally the **basic structural
stage** used by unit tests and the bounded CPU integration smoke. It does not claim to detect
secrets, credentials, personal data, or generated spam, and passing it alone does not approve a
source for a real training corpus.

- MF-047 owns bounded-memory web-text admission, immutable source revisions, filtering counters,
  hashed shards, Windows-safe loading, and exact resume.
- MF-051 owns code-specific license checks, secret/credential and PII policy enforcement,
  generated/vendor/minified/binary filtering, and evaluation-contamination checks.
- MF-059/MF-060 must record equivalent source-specific admission decisions for SFT data.

Until the owning task's source-specific policy and tests pass, that source remains smoke-only or
excluded.

MF-047 and MF-051 are now implemented for local manifests: SQLite-backed exact/near signatures
retain no rejected text, admission counters and algorithm versions are persisted, token shards are
atomic and hash-verified, and worker datasets store paths rather than live generators. FIM-derived
documents retain a parent-content digest so deterministic train/validation assignment is invariant
to the transform. Preparation refuses a non-empty output directory, preventing a stale signature
database or old shard from silently changing a rerun. Real source approval still requires a reviewed manifest and the home-GPU release
corpus gate; the implementation does not make a legal determination.

## Storage and publication

Corpora, caches, checkpoints, credentials, and machine-specific paths are never committed. Published model cards must identify source families, versions, licenses, mixture proportions, filtering, known limitations, and any source that cannot be redistributed.

## Review

Adding or changing a source is a recorded project decision. When terms are ambiguous, the source remains excluded until the project owner approves it. This document is an engineering policy, not legal advice.
