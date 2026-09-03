# MF-070 pre-work: Global NoPE quality test — RTX 2070 Super

Environment: same as `reports/mf070-pre-muon-vs-adamw.md`. Real data:
`data/shards/mf064-150m-train` (train + validation). Model: `configs/150m-modern.toml`
(rope arm, unmodified) vs a scratch copy with only `global_position_encoding` changed to
`"none"` (valid: `ModelConfig` only permits `"none"` when `attention_pattern == "hybrid"`,
which 150m-modern already is). Fixed LR 3e-4 (AdamW), batch=2, FP16, seed 42, ~10.23M
tokens/arm. Raw records: `artifacts/mf070-nope/nope-*/run.json` and `validation.json`.

This closes the gap `labs/07_rope_vs_global_nope.py`'s own docstring flags: *"This lab
proves flag isolation, not quality... Whether NoPE is better can only be answered by
matched training runs, and this script deliberately makes no such claim."* This is that
matched training run.

## Results (in-distribution, sequence_length=1024)

| Global position encoding | Train loss | Val CE | Val PPL | Val BPB | Tokens/s | Peak reserved VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rope (default) | 5.234 | **5.145** | **171.6** | 1.731 | 4,249 | 5.63 GB |
| none (NoPE, global layers only) | 5.258 | 5.158 | 173.8 | 1.735 | 4,252 | 5.63 GB |

RoPE stays on every local layer in both arms — `global_position_encoding` only affects the
one global layer in this 20-layer, Local/Local/Local/Global hybrid schedule.

## Findings

1. **RoPE slightly outperforms NoPE at this scale, in-distribution.** Validation
   perplexity 171.6 vs 173.8 — a small but consistent gap (also visible in train loss,
   5.234 vs 5.258). Not dramatic, but real and in RoPE's favor.
2. **Throughput and VRAM are effectively identical** between the two (4,249 vs 4,252
   tok/s, same 5.63 GB) — removing RoPE from one layer has no measurable cost/benefit on
   its own at this scale.

## Long-context extrapolation follow-up

The result above only tests in-distribution quality at the trained `sequence_length=1024`
— it cannot speak to NoPE's actual research claim, which is about **length
generalization** (quality at sequence lengths *beyond* training length), not general
in-distribution quality. See `reports/mf070-nope-long-context-extrapolation.md` for that
follow-up, run against these same two checkpoints.

## What this does not settle

- Single seed, ~10.23M tokens, far short of the frozen 3B-token target — exploratory, not
  scale-representative.
- This is a **narrow, config-isolated** test (NoPE on one global layer of twenty) —
  matches this project's own frozen scope (RoPE-everywhere by default, global NoPE is an
  explicit experiment, never a wholesale RoPE replacement), not a general "RoPE vs NoPE"
  claim at architecture scale.

**Conclusion: no evidence from this pass supports enabling NoPE by default. RoPE remains
the correct choice for MF-070's 350M run.**
