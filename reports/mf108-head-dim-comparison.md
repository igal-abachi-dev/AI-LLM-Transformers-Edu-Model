# MF-108: real bounded comparison — head_dim 64 (baseline) vs. 96 vs. 128

Real measurements via `train/pretrain.py` + `scripts/eval_checkpoint.py`, `150m-modern.toml`
(with the current frozen `layer_norm_scaling=true`/`gated_attention=true` defaults), RTX 2070
Super, FP16, `data/shards/mf064-150m-train`, seed 42, 5,000 updates / 10,230,000 tokens per arm
(head_dim=128 is the exception — see below). Only `head_dim_override` differs between arms;
`n_heads`/`n_kv_heads` stay at 12/4 throughout, so Q/K/V/`out_proj` widen without changing head
*count*, a real, disclosed parameter/compute increase, not a matched-parameter comparison.

## Real results

| Arm | Params | Δ params | Peak VRAM | Tokens/s | Cross-entropy | Perplexity | Bits/byte | Δ BPB vs. 64 |
|---|---|---|---|---|---|---|---|---|
| 64 (current default) | 138,630,640 | — | 5,158.9 MB | 4,057.9 | 5.100674 | 164.13 | 1.71589 | — |
| **96** | 154,360,560 | +11.35% | ~7,737 MB | ~3,106.8 | **5.077986** | **160.45** | **1.70826** | **−0.4447% (real win, ~9.7x noise floor)** |
| 128 | 170,090,480 | +22.69% | ~7,440-7,739 MB | ~625 | *(run abandoned before completion)* | | | |

`head_dim=64`'s numbers are the real, already-existing `mf097-packing/ribbon/final` checkpoint
(reused as the shared baseline for this and [[MF-107]]'s own comparison, not retrained). `head_dim=128`
was intentionally stopped partway through (a real, deliberate `KeyboardInterrupt`, not a crash) once
its real cost became clear — no held-out quality number exists for it, on purpose; see the
"Why 128 was abandoned" section below.

## Why 128 was abandoned before completion

Live-monitored (`nvidia-smi`, checkpoint timestamps) rather than measured via a completed benchmark
record, since the run never reached `final/`: peak VRAM sat at 90-95% of this card's 8,192 MB physical
total (~7,440-7,739 MB, against the baseline's 5,158.9 MB — a real **+44.6%** VRAM increase from
widening `head_dim` alone), and real throughput collapsed to ~625 tok/s (~6.5x slower than baseline's
4,057.9 tok/s) — far more than pure compute scaling from doubling `head_dim` could plausibly explain.
This matches this project's own already-documented "8GB ceiling silently pages rather than OOMs"
finding (`AGENTS.md`; directly measured once before in [[MF-050]]'s real 8K-context test). Two
alternative explanations (this session's own same-day code changes; GPU contention from a
concurrently-running process) were checked and ruled out before attributing this to VRAM pressure —
see [[MF-108]]'s own backlog entry for the full reasoning.

## Why 96 is a genuinely different, real result

Unlike 128, `head_dim=96` also lands close to the same VRAM range (~7,737 MB, comparable to 128's own
number, not the ~6,310 MB a naive linear-VRAM-scaling estimate predicted) — but its real throughput
(~3,106.8 tok/s, only ~23% slower than baseline) shows nothing like 128's collapse. The VRAM-pressure
mechanism is real and directionally correct, but evidently not a simple linear threshold: something
about 128's specific memory footprint crosses a real cliff that 96's comparable-but-lower footprint
does not. Not independently diagnosed further (would need real allocator-level profiling); recorded
as a measured fact, not a fully explained mechanism.

**And unlike 128, `head_dim=96` completed the full 5,000-update run and shows a real quality win**:
−0.4447% relative BPB/cross-entropy, and −2.24% relative perplexity — about **9.7x** this project's own
measured single-config noise floor (~0.046%, `reports/mf088-seed-variance.md`), comparable in
magnitude to or larger than several already-adopted findings this session (LN-scaling: −0.35%;
gated-attention: −0.67%). This is a real, single-seed, single-scale result, not a 3-seed causal claim —
but it clears this project's own established "several times the noise floor" bar for "plausibly real."

## Real disclosed cost of adopting head_dim=96

+11.35% parameters (154,360,560 vs. 138,630,640) and a real ~23% training-throughput cost (not free) —
any adoption decision needs to weigh this real quality win against that real, ongoing training-time
cost for the rest of this project's training budget, not just the parameter count.

## Limitations

Single seed, single scale (150M), single ~10.23M-token bounded budget, as with every other bounded
comparison this session. `head_dim=128`'s own quality (if any) was never measured — only its real,
disqualifying cost was. Whether `head_dim=96`'s real win would hold at 350M scale (where VRAM headroom
and the exact same paging-cliff mechanism may behave differently) is untested.
