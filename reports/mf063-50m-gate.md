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
performed — it requires a second real training run under a different data mixture and was out of
scope for this pass. This is the one remaining empirical gap before `freeze_protocol.py` could move
this protocol from `draft` to `frozen`.

## Prerequisite bug fix: `KVCache` device-index mismatch

Before any of the above could run, every cached CUDA forward pass failed immediately with
`ValueError: cache device does not match model inputs`. Root cause: `torch.device("cuda")` (no
index) and a real tensor's `torch.device("cuda", 0)` compare unequal in PyTorch, and
`LayerKVCache.allocate` was storing the former as `requested_device` while `KVCache.validate`
compared it against the latter. Fixed in `src/minifrontier/cache.py` by resolving the CUDA index at
allocation time. This blocked every documented `--device cuda` command in the project and was
invisible to CPU-only CI (CPU has no index ambiguity). Regression-covered by the new
`test_cuda_bfloat16_full_and_cached_logits_match_within_declared_tolerance` test.
