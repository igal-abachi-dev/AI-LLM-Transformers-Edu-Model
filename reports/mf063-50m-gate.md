# MF-063 home-RTX 50M FineWeb-Edu gate — real evidence

This is the first real (non-CPU, non-synthetic) run in the project's history: real pinned-revision
FineWeb-Edu data, a freshly trained real tokenizer, real CUDA training on the frozen 50M Edu
construction, and real inference evidence, all on the user's home RTX 2070 SUPER. It proves
integration and a genuine loss/validation decrease. It is **not** a model-quality claim — 2.2M
tokens on a 53M-parameter model is far below what fluent generation requires, and the sample below
demonstrates that honestly.

Environment: Windows 11, Python 3.12.10, PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 2070 SUPER
(Turing, CC 7.5, 8GB VRAM), driver 591.86. Date: 2026-08-21. Git commit
`ad47b50232650220c5b4707fcd83fef1cb5eb9b0`.

## Data pipeline

- Tokenizer corpus: 3,000 documents from `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, pinned
  revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (start=0), 14,467,506 characters.
  `scripts/train_tokenizer.py` → real 16,384-entry byte-level BPE tokenizer at `data/tokenizer`.
- Training/validation shards: 2,000 further documents from the same pinned source (start=3000, to
  avoid any overlap with the tokenizer-training text), `scripts/prepare_data.py --source fineweb-edu
  --sequence-length 1024`:
  - **train: 2,213,888 non-padding tokens**, 2,162 sequences, 3 shards — within the 1–5M target.
  - validation: 12,288 non-padding tokens, 12 sequences, 1 shard.
  - Exact/near-duplicate admission ran before the split (`minifrontier-shards-v2`, simhash-token-3gram
    near-dedup); 2,000/2,000 documents admitted, 0 rejected.

## Training gate (`train/pretrain.py --device cuda`)

Config `configs/50m-edu.toml` (53,361,152 parameters, matching the frozen MF-019 exact count),
`attention_impl=sdpa`, precision `bfloat16` (`deterministic_algorithms=false` — the default,
non-strict mode; not explicitly requested), seed 42, batch_size 2, 1,081 updates (one full epoch over
the packed train shard), warmup 40, cosine decay 3e-4 → 3e-5.

| Metric | Value |
| --- | ---: |
| Loss, blind start (ln 16384 ≈ 9.7) → final | 9.7 → **6.163** |
| Train tokens consumed | 2,211,726 |
| Wall time | 313.4s |
| Throughput | 7,058 tokens/s |
| Peak allocated VRAM | 3.07 GB / 8.59 GB |
| Compile | not used for the gate itself (`torch.compile` was evaluated separately — see `reports/mf050-rtx2070s-profile-matrix.md`, no benefit found on this GPU) |

Raw record: `artifacts/mf063-50m-gate/run.json`.

### Validation (before vs after training)

Computed with `evaluate_token_batches` against the full real validation shard (12 sequences, 12,276
predicted tokens after masking), using the exact same-seed untrained model as the "before" baseline:

| | Cross-entropy | Perplexity | Bits/byte |
| --- | ---: | ---: | ---: |
| Before training | 10.150 | 25,580 | 3.308 |
| After training | **6.110** | **450** | **1.991** |

Raw record: `reports/mf063-50m-gate-validation.json`.

### Exact resume, on real hardware

A separate 300-update run on the same real data was genuinely interrupted mid-run (via `TaskStop`,
not a graceful stop) after checkpoint-150, then resumed from that checkpoint with the identical CLI
config to reach update 300. Two independent interruption points (checkpoint-100 in one attempt,
checkpoint-150 in another) both resumed to the **exact same final loss** (`6.713391304016113`),
which is strong real-hardware corroboration of exact resume, though not a formal weight-by-weight
tolerance diff against an uninterrupted control on this specific 50M/BF16/CUDA configuration (that
rigor exists as a deterministic CPU/FP32 unit test — `test_checkpoint_resume_is_exactly_equal_to_uninterrupted_training`).

**Bug found and fixed along the way**: `train/pretrain.py`'s `tokens_per_second` divided the
checkpoint's *cumulative* token count by only the *resumed process's* wall time, silently inflating
throughput after any resume (measured 10,388 tok/s before the fix vs. a corrected, sane 6,728 tok/s
after — both from the identical resumed workload). Fixed to track only tokens processed by the
current process. This would have corrupted throughput evidence for every multi-session MF-064/065
150M run without the fix.

### Inference evidence

- Real cached BF16 greedy sample from the trained checkpoint (`scripts/sample.py`,
  `--precision bfloat16 --temperature 0.0`): *"The history of science shows that the first time of
  the same time. The first time is the same time of the same time..."* — repetitive and incoherent,
  as expected for a checkpoint this far from convergence. Reported honestly as **undertrained**, not
  a quality result.
- Real release export (`scripts/export.py`) from this actual trained checkpoint succeeded and passed
  the same load-test path `tests/test_release.py` exercises on toy fixtures — real, non-toy evidence
  that the MF-067 export/audit mechanics work end to end.
- Prefill/decode timing, TTFT, and cache-dtype evidence for this preset are in
  `reports/mf050-rtx2070s-profile-matrix.md` (50M-edu rows).
- `bounded-local-cache parity`: not applicable — 50m-edu uses full attention with no local/ring cache
  layers; this only applies to the Modern preset.

### Standard evaluation status

`scripts/eval.py --run-harness --limit 20` (lm-eval 0.4.12, newly installed — see below) against the
real exported checkpoint, ARC-Easy/HellaSwag/PIQA, 0-shot, n=20 samples per task:

| Task | acc | acc_norm | Chance baseline |
| --- | ---: | ---: | ---: |
| arc_easy | 0.20 | 0.15 | ~0.25 (4-choice) |
| hellaswag | 0.30 | 0.30 | ~0.25 (4-choice) |
| piqa | 0.60 | 0.45 | ~0.50 (2-choice) |

All scores are within noise of chance at n=20, consistent with a 2.2M-token checkpoint — expected,
not a regression. Raw record: `reports/mf063-50m-gate-lmeval.json`. (`lm-eval` was declared in
`pyproject.toml`'s `eval` dependency group but not yet installed in this environment; installed via
pip directly since `uv` was not on PATH in this session — no `pyproject.toml`/`uv.lock` changes were
made.)

### Not done in this pass

**Matched FIM follow-up** (an MF-053-style normal-vs-FIM comparison rerun on this real gate) was not
performed in this original pass — it required a second real training run under a different data
mixture. This was the one remaining empirical gap before `freeze_protocol.py` could move this
protocol from `draft` to `frozen`. **Update (2026-08-26): now done — see the "MF-063 FIM follow-up"
section below, which also freezes the protocol.**

## Prerequisite bug fix: `KVCache` device-index mismatch

Before any of the above could run, every cached CUDA forward pass failed immediately with
`ValueError: cache device does not match model inputs`. Root cause: `torch.device("cuda")` (no
index) and a real tensor's `torch.device("cuda", 0)` compare unequal in PyTorch, and
`LayerKVCache.allocate` was storing the former as `requested_device` while `KVCache.validate`
compared it against the latter. Fixed in `src/minifrontier/cache.py` by resolving the CUDA index at
allocation time. This blocked every documented `--device cuda` command in the project and was
invisible to CPU-only CI (CPU has no index ambiguity). Regression-covered by the new
`test_cuda_bfloat16_full_and_cached_logits_match_within_declared_tolerance` test.

## MF-063 FIM follow-up (2026-08-26) — closes the last empirical gap before freezing

This is the "not done in this pass" item above, now run for real on the same RTX 2070 SUPER. It is a
**bounded engineering comparison**, not an effect-size claim (`scripts/compare_fim.py` itself labels
its output `"status": "bounded_engineering_comparison"`, `"quality_claim": false`) — the corpus below
is small enough that both arms partly memorize it, so the interesting result is that the pipeline is
correct and the two arms are genuinely matched, not that either loss number says something about
FIM's effect at production scale (that requires the real MF-053-scale-or-larger run this protocol
still owes once MF-064/065 exist).

### Code corpus and provenance

Unlike FineWeb-Edu, this needed a *permissively licensed code* source (`PERMISSIVE_CODE_LICENSES` in
`src/minifrontier/data.py`). Rather than reach for a new external dependency with its own licensing
review, this run uses **this repository's own Apache-2.0 Python source** (`src/minifrontier/*.py`,
`scripts/*.py`, `train/*.py` — 61 files) as the code manifest: `source` = this repo's real GitHub
remote (`https://github.com/igal-abachi-dev/AI-LLM-Transformers-Edu-Model`), `revision` = the real
commit SHA this run executed at (`4986326304f399c1aaf896e1856753ebeced36df`), `license` =
`Apache-2.0` (see `LICENSE` — already the project's own license, so provenance is unambiguous and
requires no separate legal review).

`scripts/prepare_code.py` admitted 57/61 files (1 rejected as `generated`, 3 as `personal_data` —
real hits from the admission screen, not synthetic test cases; the screen also checked the corpus
against `eval/fixtures/code_fim_v1.jsonl` via freshly computed exact/simhash signatures
(`data/mf063-fim-followup/evaluation-signatures.json`, built with the same `normalized_sha256`/
`simhash64` functions `admit_documents` itself uses) — zero evaluation-fixture overlap found, as
expected since the fixtures are small original synthetic functions distinct from this project's real
source.

### FIM mixing and packing

`scripts/apply_fim.py --seed 42` produced two manifests from the same 57 approved documents: `--rate
0.0` (0/57 transformed — a true unchanged baseline) and `--rate 0.15` (7/57 transformed, ≈12% at this
sample size, consistent with the frozen default). Both were packed independently with
`scripts/prepare_data.py` using the project's one frozen tokenizer (`data/tokenizer`,
`sha256=6991f9356bc3ce5661292fe14b8980c0b2188ec84b6389c9b760dadd86aba402`) at `sequence_length=1024`:
both arms packed to **exactly 153 train sequences / 156,672 non-padding tokens** — FIM rearranges
text in place rather than adding or removing characters, so packing boundaries landed identically
even though the actual token content differs (confirmed by differing `tokens_sha256` but identical
`counts_sha256` in each `metadata.json`).

### Training comparison

`scripts/compare_fim.py --config configs/50m-edu.toml --seed 42 --batch-size 2 --updates 300 --device
cuda` (real CUDA run, ~73 seconds wall time). Both arms are freshly initialized from the *same*
starting weights (the script copies `baseline.state_dict()` into the FIM model before training) and
consumed **identically 613,800 tokens** each (the script itself raises if they don't match — they
matched by construction). `training_config.precision="auto"` resolved to real Tensor Core **FP16**
with `GradScaler` on this hardware (`grad_scaler_state.scale=32768.0` in the output), per MF-075's
hardware-aware `"auto"` policy — not emulated BF16.

| Arm | Final loss | Final grad norm | Consumed tokens |
| --- | ---: | ---: | ---: |
| Baseline (no FIM) | **1.612** | 3.122 | 613,800 |
| FIM (rate 0.15) | **1.668** | 3.192 | 613,800 |

Raw record: `artifacts/mf063-fim-gate/comparison.json`.

**Reading this honestly**: both losses are low because 153 sequences repeated over 300 updates
(≈3.9 cycles) at batch_size=2 is heavy repetition of a tiny corpus — this is closer to memorization
than generalization, expected and fine for a pipeline-correctness check. The FIM arm's slightly
higher final loss and gradient norm is the real, visible **cost** `compare_fim.py`'s own docstring
warns about: part of its identical token budget went to learning the `<|fim_prefix|>`/`<|fim_suffix|>`/
`<|fim_middle|>` rearrangement protocol on 7 of the 57 documents, which is a real distributional
shift the plain-continuation arm never has to absorb. This matches the expected qualitative
direction (FIM costs some ordinary-continuation budget) without claiming a production-scale effect
size — that claim needs the larger, longer run this protocol is now freezing itself to eventually
require.

### Protocol frozen

With this evidence in hand, `scripts/freeze_protocol.py` produced a valid `status="frozen"` protocol:

```
scripts/freeze_protocol.py --status frozen \
  --tokenizer-sha256 6991f9356bc3ce5661292fe14b8980c0b2188ec84b6389c9b760dadd86aba402 \
  --data-mixture-id fineweb-edu-v1-pinned-87f09149 \
  --target-tokens 3000000000 --batch-tokens 2048 --sequence-length 1024 \
  --optimizer adamw --learning-rate 3e-4 --seeds 42 --evaluation-interval 500 \
  --evidence reports/mf063-50m-gate-validation.json reports/mf063-50m-gate-lmeval.json \
             artifacts/mf063-fim-gate/comparison.json artifacts/mf063-50m-gate/run.json \
  --output artifacts/mf063-protocol/frozen.json
```

`target_tokens=3,000,000,000` is kept at the full frozen V1 target (not lowered to match this
session's smaller planned MF-064/065 budget) — freezing declares the **recipe** the eventual 3B-token
runs are meant to follow, backed by this measured small-scale evidence that the recipe is sound, not
a claim that 3B tokens have already been trained. `undertrained_approved` correctly stays `false`
here: `TrainingProtocol.__post_init__` (`src/minifrontier/release.py`) only requires that flag when
`target_tokens < 3_000_000_000`, which does not apply to the frozen protocol itself. The user has
separately decided MF-064/065 will actually run at a smaller, explicitly-undertrained token budget
"for now" — that is a property of those runs' own evidence/labeling when they happen, not of this
frozen recipe. Raw record: `artifacts/mf063-protocol/frozen.json`.

### Reproducing this pass

Working files live under `data/mf063-fim-followup/` (gitignored, like all of `data/`): a small
one-off `build_manifest.py` (documents its own logic inline) builds the raw code manifest and
evaluation-signatures JSON from this repo's own source at the commit above; everything downstream
uses only the existing project CLIs.
