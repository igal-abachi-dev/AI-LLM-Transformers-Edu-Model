# MF-070 pre-work: corrected local-window re-test — RTX 2070 Super

Environment: same as `reports/mf070-pre-muon-vs-adamw.md`. Real data:
`data/shards/mf064-150m-train` (train + validation). Model: `configs/150m-modern.toml`
with only `local_window` changed (scratch copies; the tracked preset is untouched), fixed
LR 3e-4 (AdamW), batch=2, FP16, seed 42, ~10.23M tokens/arm (5,000 updates,
`sequence_length=1024`). Raw records: `artifacts/mf070-local-window/window-*/run.json`
and `validation.json`.

## Why this re-test exists

The project's original local-window pre-work (`tasks/backlog.md`, 2026-08-27) compared
`local_window=512` vs `local_window=1024` — but `sequence_length=1024` was used for that
training data too. `src/minifrontier/masking.py`'s window rule
(`key_position >= query_position - window_size + 1`) becomes a **complete no-op** whenever
`window_size >= sequence_length`: every already-causal key automatically satisfies it. So
the "1024" arm was not "a wider local window" — it was **full/global attention with no
local restriction at all**. This re-test uses window values that are all genuinely
`< sequence_length=1024`, so the local/global distinction is actually exercised in every
arm.

## Results

| Local window | Train loss | Val CE | Val PPL | Val BPB | Tokens/s | Peak reserved VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5.241 | 5.139 | 170.5 | 1.729 | 4,353 | 5.63 GB |
| 256 | 5.241 | 5.140 | 170.8 | 1.729 | 4,350 | 5.63 GB |
| 512 | 5.234 | 5.145 | 171.6 | 1.731 | 4,320 | 5.63 GB |

For reference, the original (degenerate) "1024" arm — mathematically equivalent to no
window restriction — scored validation CE 5.147 / PPL 171.9.

## Findings

1. **Local window size makes no measurable difference at this scale and token budget,**
   across a real, genuinely-restrictive 4x range (128 vs 512) — validation perplexity
   spans only 170.5 to 171.6, well within single-seed noise.
2. **This holds even including the degenerate "no restriction" case.** The original
   flawed test's "1024" arm (171.9 PPL) is barely different from the corrected 512 arm
   (171.6 PPL) here. Put together, the full picture across restricted (128/256/512) *and*
   effectively-unrestricted (1024+) windows is: **local windowing, whether present or
   absent, doesn't measurably affect quality at this 150M-parameter, ~10M-token scale.**
3. **Throughput and VRAM are also flat across window sizes** (4,320–4,353 tok/s, 5.63 GB
   reserved throughout) — consistent with MF-050's own finding that eager (uncompiled)
   FlexAttention materializes the full score matrix regardless of window size, so window
   size doesn't change per-step compute cost in this project's current eager execution
   path.

## What this does not settle

- This says nothing about whether local windowing matters for the **KV-cache memory**
  savings the hybrid architecture is designed to provide at inference time (MF-050's own
  ~3.4x measured ring-cache byte savings) — that benefit is real and already measured
  separately; this report is about training-time *quality*, not inference-time *memory*.
- **1024 and 2048 arms were deliberately not added** to this re-test: both would be
  mathematically identical no-op cases at `sequence_length=1024` (the training data's
  actual packed length), providing no new information over what the original test's
  degenerate "1024" arm and this test's own 512 arm already show.
- A genuinely informative test of whether local windowing's *quality* effect shows up at
  all would need training at a **longer sequence length** (e.g. 2048) so window values
  stay meaningfully restrictive relative to it — that is Priority 2b from the approved
  MF-070 pre-work plan, explicitly deferred (needs a new data-packing pass).
- Single seed, ~10.23M tokens, far short of the frozen 3B-token target — exploratory, not
  scale-representative.

**Conclusion: `local_window=512` (the frozen default) is not starving quality relative to
narrower windows, and there is no evidence from this pass to justify changing it before
MF-070's 350M run.**
