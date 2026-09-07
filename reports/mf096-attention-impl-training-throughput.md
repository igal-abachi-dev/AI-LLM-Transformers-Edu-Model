# MF-096 — real training-throughput comparison: auto vs forced SDPA vs forced manual

## Commands

```
./.venv/Scripts/python.exe train/pretrain.py --config configs/150m-modern.toml \
  --train-shards data/shards/mf064-150m-train/train --updates 200 --batch-size 2 \
  --seed 42 --device cuda --no-checkpoint --attention-impl <auto|sdpa|manual> \
  --output artifacts/mf096-attention-impl/<label>
```

RTX 2070 Super, FP16 (`auto` precision), `150m-modern.toml` unmodified (hybrid, `local_window=512`,
`n_layers=20`), real data (`data/shards/mf064-150m-train/train`), 200 updates each (409,200 real
tokens/arm), 2026-09-07/08. Raw records: `artifacts/mf096-attention-impl/{auto,sdpa,manual}/run.json`.

## Why this run exists

A fourth round of external feedback flagged "your eager local FlexAttention path is painfully
slow" as a real problem worth fixing. That specific claim is not new — `reports/mf050-rtx2070s-profile-matrix.md`
already found eager FlexAttention 6.4x slower than manual attention **for prefill**, and MF-078
already tried and ruled out compiling FlexAttention directly (a genuine PyTorch Inductor bug,
reproduced on both CPU and this GPU). But re-reading those reports found a real, previously
unexamined gap: every existing comparison used **inference** (prefill/decode), never **training**
throughput, and `attention.py` already implements a correct, already-used-for-global-layers SDPA
path with an explicit banded window mask for local layers (lines ~405-418) — a real, already-built
alternative to FlexAttention that had simply never been benchmarked for training. The hypothesis
going in: since MF-077's own report found "manual is consistently slower than SDPA... on
full-attention models," and SDPA's fused kernel usually beats a from-scratch matmul+softmax
implementation, forced SDPA might beat both manual and the current default for training throughput.

## Real result — the hypothesis was wrong; the current default already wins

| `attention_impl` | tokens/s | relative to `auto` | peak reserved VRAM | train loss (200 updates) |
|---|---:|---:|---:|---:|
| **auto** (current default: flex-local, sdpa-global) | **4,306.4** | — | **5.63 GB** | 7.154 |
| sdpa (forced, all layers) | 2,109.7 | **2.04x slower** | 7.11 GB (+1.48 GB) | 7.169 |
| manual (forced, all layers) | 1,290.3 | **3.34x slower** | 7.87 GB (+2.24 GB) | 7.124 |

Loss values are consistent across all three (differences are step-to-step noise at 200 updates,
not a correctness signal either way — all three implementations compute the same mathematical
attention, verified separately by this project's own existing parity tests). The result that
matters here is throughput and memory, and on both counts **the current `auto` default is the
clear winner**, not a compromise waiting to be beaten:

- Forcing SDPA everywhere (including local layers, via the explicit banded mask) is **more than
  2x slower** than the current mixed default, and uses **more** memory, not less — consistent with
  a known PyTorch SDPA behavior: supplying a custom `attn_mask` tensor (rather than `is_causal=True`
  or no mask) disables the fastest fused/Flash-Attention-style kernel path, forcing a slower
  "efficient" or "math" backend, and the explicit `[Sq, Sk]` mask itself costs real (if modest at
  this project's `sequence_length`) memory that the implicit `is_causal` description avoids
  entirely.
- Forcing manual attention everywhere is slower still (3.34x), as expected for a path with no
  fused kernel at all — consistent with MF-077's own finding that manual is slower than SDPA on
  full-attention models, now confirmed to hold for training on this hybrid preset too, not just
  inference.

## What this settles

**The premise of this task's hypothesis does not hold.** MF-050's real finding — manual beats
eager FlexAttention *for prefill* — does not generalize to training throughput, where the
project's own existing `"auto"` heuristic (FlexAttention for local layers, SDPA for global layers)
already outperforms every tested alternative by a wide margin. There is no free training-throughput
win sitting unused in this codebase; the current default was already, empirically, the right
choice for training on this hardware. **No change to `ModelConfig.attention_impl_for_layer`'s
`"auto"` resolution rule is warranted.** MF-070's real 350M run should proceed with
`attention_impl="auto"` (the config default, unmodified) exactly as already planned — this
investigation confirms rather than changes that plan.

## Caveats

- Single seed, 200-update bounded smoke run (409,200 tokens/arm) — enough to get past initial
  CUDA warmup and produce a stable throughput reading, not a long-run statistical comparison.
- FP16 precision (`auto` on this Turing GPU, per this project's own hardware-aware policy) — not
  retested under FP32/BF16; the relative ordering is expected to hold but was not reverified under
  other precisions.
- Peak VRAM differences are real and measured, but this comparison used `batch_size=2` (this
  project's established safe batch size for this preset on this hardware) — the memory delta
  between `auto` and `sdpa` (+1.48 GB) could matter more at a batch size closer to this card's
  ceiling, though `auto` already being both faster and lower-memory means there is no scenario
  found here where `sdpa` or `manual` would be preferable.
