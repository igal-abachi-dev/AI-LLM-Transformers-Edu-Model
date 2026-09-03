# MF-070 pre-work: real AdamW-vs-Muon GPU comparison — RTX 2070 Super

Environment: Windows 11, Python 3.12.10, PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 2070 SUPER
(Turing, CC 7.5, 8 GB VRAM). Date: 2026-09-02/03. Real data: `data/shards/mf064-150m-train`
(train split for training, `validation` split for the post-hoc quality check — same real
FineWeb-Edu pool used by the local-window pre-work). Model: `configs/150m-modern.toml`
(138,446,080 parameters), unmodified. Real command output only; raw per-arm records:
`artifacts/mf070-pre-muon-vs-adamw/*-result.json` (training) and `*-validation.json`
(quality). Six arms, single seed 42, ~10.23M tokens each (5,000 updates, batch=2,
sequence_length=1024, matching the local-window pre-work's scale).

This fulfills MF-057's own deferred commitment: its status note explicitly says *"the
matched-token CPU engineering sweep... records partitioning/loss/throughput without an
optimizer-quality claim. The post-M10 RTX rerun owns VRAM, variance, and scale
conclusions."* This is that rerun.

## Results

| Optimizer | LR | Train loss | Val CE | Val PPL | Val BPB | Tokens/s | Wall (s) | Peak reserved VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 1e-4 | 5.475 | 5.380 | 217.1 | 1.810 | 4,312 | 2,372 | 5.63 GB |
| AdamW | 3e-4 | 5.225 | 5.137 | 170.3 | 1.728 | 4,267 | 2,398 | 5.63 GB |
| **AdamW** | **1e-3** | 5.240 | **5.130** | **169.0** | 1.726 | 4,409 | 2,320 | 5.63 GB |
| Muon | 3e-4 | 5.505 | 5.417 | 225.1 | 1.822 | 1,898 | 5,389 | 5.08 GB |
| Muon | 1e-3 | 5.141 | 5.045 | 155.3 | 1.697 | 1,903 | 5,376 | 5.08 GB |
| **Muon** | **3e-3** | 4.902 | **4.803** | **121.9** | 1.616 | 1,886 | 5,423 | 5.08 GB |

Bold rows are each optimizer's best-of-3 arm, per this project's "each judged at its own
best LR" discipline (a single shared LR would unfairly bias the comparison — Muon's update
is normalized, so its natural LR scale is unrelated to AdamW's).

## Findings

1. **Muon's best arm clearly beats AdamW's best arm on quality-per-token.** Validation
   perplexity 121.9 vs 169.0 — a real ~28% relative improvement, at the identical token
   budget, same data, same seed. This is not a small or ambiguous gap.
2. **Muon is consistently ~2.3x slower than AdamW on this hardware, independent of LR.**
   All three Muon arms measured 1,886–1,903 tok/s; all three AdamW arms measured
   4,267–4,409 tok/s. This is the real, honest cost side of Muon's quality win — the
   Newton-Schulz orthogonalization step this project's implementation uses
   (`torch.optim.Muon`, first-party, `ns_steps=5` default) adds real per-step compute that
   AdamW's plain elementwise updates don't pay.
3. **Muon uses meaningfully less peak VRAM** (5.08 GB vs 5.63 GB reserved) — AdamW's
   per-parameter first/second moment buffers on the muon-eligible 2-D matrices are absent
   under Muon (those matrices use Newton-Schulz orthogonalization instead, no persistent
   optimizer state of that size), while AdamW still applies its usual state to everything.
   A real but secondary benefit next to the throughput cost.
4. **Muon strictly improved with every higher LR tested** (3e-4 → 1e-3 → 3e-3, worse →
   better → best) — the sweep did not find where Muon's LR curve peaks or begins to
   destabilize. `3e-3` was the top of the tested range, not a confirmed optimum. A
   follow-up arm at a higher LR (e.g. `1e-2`) is planned to find the real ceiling before
   any final optimizer recommendation.
5. **Optimizer choice affects training speed only, not inference speed.** The optimizer
   is not part of the trained model — it only computes weight updates during training.
   Once training stops, Muon-trained and AdamW-trained weights are indistinguishable to
   `sample.py`/`chat.py` in terms of runtime cost. Muon's 2.3x slower training is a
   one-time cost per training run, not a permanent tax on the deployed model.

## What this does not settle

- **Quality-per-wall-clock-time is still open.** This comparison matches *tokens*, not
  wall-clock time. Given Muon takes ~2.3x longer to consume the same tokens, whether it
  would still win if AdamW were given the same *wall-clock budget* (and therefore ~2.3x
  more tokens) is a genuinely different, unanswered question — not addressed by this pass.
- **Muon's LR ceiling is unconfirmed** (finding 4 above) — a real optimum may lie above
  3e-3, or the curve may already be near its peak. Untested.
- **Single seed, ~10.23M tokens** — far short of the frozen 3B-token target. This is an
  exploratory engineering comparison (matching `compare_optimizers.py`'s own
  `"status": "bounded_engineering_comparison"` framing), not a scale-representative claim.
- **Speed-improvement research exists but is untested here** (reduced `ns_steps`,
  `torch.compile` on just the optimizer step, hardware-aware Newton-Schulz variants like
  Gram Newton-Schulz) — real 2026 research on exactly this bottleneck, planned as
  follow-up, not run in this pass.

No optimizer-switch decision is made in this report. It supplies the real evidence MF-057
was always missing; the actual choice for MF-070's 350M run (and beyond) is deferred to a
follow-up pass that also probes Muon's LR ceiling and speed-optimization options.
