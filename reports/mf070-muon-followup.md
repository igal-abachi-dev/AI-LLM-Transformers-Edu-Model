# MF-070 pre-work: Muon follow-up — LR ceiling, speed optimizations, wall-clock-matched AdamW

Environment: same as `reports/mf070-pre-muon-vs-adamw.md` (Windows 11, Python 3.12.10,
PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 2070 SUPER, Turing, CC 7.5, 8 GB VRAM). Dates:
2026-09-03 (six-arm round) and 2026-09-04 (decisive hybrid arm, see finding 6). Real data:
`data/shards/mf064-150m-train` (train split for training, `validation` split for the
post-hoc quality check). Model: `configs/150m-modern.toml` (138,446,080 parameters),
unmodified. Real command output only; raw per-arm records were captured in
`artifacts/mf070-muon-followup/*-result.json`/`*-validation.json` and
`artifacts/mf070-muon-hybrid-decision/*-result.json`/`*-validation.json` (both directories
deleted after their numbers were recorded here, per this project's established
storage-hygiene practice). Single seed 42 throughout, matching this project's established
exploratory-comparison discipline.

This is the direct follow-up to `reports/mf070-pre-muon-vs-adamw.md`, which closed with five
explicit open questions: Muon's LR ceiling was unconfirmed (3e-3 was the top of the tested
range), `ns_steps` reduction was untested, `torch.compile` on the optimizer step was
untested, a hardcoded-bfloat16 precision issue in `torch.optim.Muon`'s internals had just
been discovered, and quality-per-*wall-clock-time* (as opposed to quality-per-*token*) was
explicitly flagged as unanswered. The six-arm round below closes all five in one pass.
A sixth, decisive arm (finding 6) was added the next day after the user asked whether
combining the two real speed wins (FP32 Newton-Schulz + `ns_steps=3`) at Muon's best known LR
could close the wall-clock-matched gap against AdamW — first answered by extrapolation, then
settled by a real, direct test at the user's request.

## Results

Six arms, ~10.23M tokens each (5,000 updates, batch=2, sequence_length=1024) except the
wall-clock-matched AdamW arm (11,700 updates, ~23.94M tokens — see below).

| Arm | Optimizer | LR | ns_steps | ns_dtype | compile | Val CE | Val PPL | Val BPB | Tokens/s | Wall (s) | Peak reserved VRAM |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lr-ceiling-1e-2 | Muon | 1e-2 | 5 | bf16 | no | **4.707** | **110.75** | 1.584 | 1,885.5 | 5,425 | 5.08 GB |
| ns3-at-best-3e-3 | Muon | 3e-3 | 3 | bf16 | no | 4.844 | 126.99 | 1.630 | **2,429.1** | 4,211 | 5.08 GB |
| compile-step-3e-3 | Muon | 3e-3 | 5 | bf16 | **yes** | 5.964 | 389.23 | 2.006 | 1,847.0 | 5,539 | 4.98 GB |
| stacked-ns3-compile | Muon | 3e-3 | 3 | bf16 | **yes** | 4.921 | 137.21 | 1.656 | 2,422.7 | 4,223 | 4.98 GB |
| **fp32-ns-3e-3** | Muon | 3e-3 | 5 | **fp32** | no | 4.800 | 121.50 | 1.615 | 2,277.5 | 4,492 | 5.08 GB |
| adamw-wallclock-matched | AdamW | 1e-3 | — | — | no | **4.504** | **90.38** | 1.515 | 4,273.5 | 5,602 | 5.63 GB |

For reference, the prior report's baselines: Muon lr=3e-3/ns_steps=5/bf16/no-compile scored
val CE 4.803 / PPL 121.9 at ~1,886 tok/s, 5,423s wall, 5.08 GB (this is `fp32-ns-3e-3`'s
direct control arm — identical config except the Newton-Schulz precision). AdamW
lr=1e-3/5,000-updates (matched *tokens*, not matched *time*) scored val CE 5.130 / PPL 169.0
at ~4,409 tok/s, 2,320s wall.

## Findings

1. **Muon's LR ceiling is still not found.** lr=1e-2 continues the monotonic improvement
   from the original sweep (3e-4 → 1e-3 → 3e-3, worse → better → better) with PPL 110.75 —
   clearly better than 3e-3's 121.9. Speed and VRAM are unchanged from 3e-3 (1,885.5 vs 1,886
   tok/s, same 5.08 GB), confirming LR doesn't affect per-step compute cost. A real optimum
   for this scale/token-budget remains untested above 1e-2.
2. **`ns_steps=3` is a genuine, honest speed/quality tradeoff — not free.** A real ~29%
   throughput gain (2,429 vs 1,886 tok/s) comes with a real ~4% relative PPL cost (126.99 vs
   121.9 at the same lr=3e-3). Useful if wall-clock is the binding constraint and a small
   quality loss is acceptable; not a strict improvement.
3. **`torch.compile` on Muon's optimizer step is broken on this build — avoid it entirely.**
   Two arms tested it (`compile-step-3e-3` at ns_steps=5, `stacked-ns3-compile` at
   ns_steps=3). Neither gave any real speedup — `compile-step-3e-3` was actually slightly
   *slower* than its uncompiled control (1,847 vs 1,886 tok/s), and `stacked-ns3-compile`
   matched its uncompiled control almost exactly (2,422.7 vs 2,429.1 tok/s), i.e. compiling
   added nothing. Worse, quality was corrupted in both arms relative to their respective
   uncompiled controls: `compile-step-3e-3` catastrophically (PPL 389.2 vs 121.9, +219%) and
   `stacked-ns3-compile` moderately (PPL 137.2 vs 126.99, +8%). The severity is
   config-dependent and not fully understood — plausibly a Dynamo tracing/guard interaction
   with the stateful bound-method momentum buffer across many real optimizer steps, since an
   earlier isolated 10-step synthetic smoke test (tiny random matrices, no real training loop)
   showed a clean ~16% speedup with no correctness issue. That smoke test was too short to
   expose whatever this is. **Practical conclusion, independent of root cause: do not use
   `torch.compile` on `torch.optim.Muon.step()` on this PyTorch build.** This is unrelated to
   MF-078's separate, already-closed finding that `torch.compile(flex_attention)` fails to
   lower on this build (`InductorError: LoweringException: SubgraphLoweringException`,
   independently reconfirmed earlier this session) — that failure is a hard compile error;
   this one silently produces a runnable but corrupted result, which is arguably worse.
   Together, both findings suggest `torch.compile` is broadly unreliable in this project's
   current PyTorch/CUDA/Windows combination, not just unhelpful in one specific place.
4. **FP32 Newton-Schulz is a clean, unambiguous win — the standout finding of this pass.**
   Reading the installed `torch.optim._muon` source directly (not from memory) showed its
   Newton-Schulz iteration unconditionally casts to `bfloat16`
   (`ortho_grad = grad.bfloat16()`), with no public parameter to change it. This project's own
   `AGENTS.md`/`reports/mf049-rtx2070s-checkpointing-benchmark.md` already measured that this
   RTX 2070 Super has no native BF16 tensor cores and that emulated BF16 runs ~2.7x slower
   than FP16 for full-model training on this hardware. A monkeypatch replacing the internal
   cast with FP32 (verified to actually intercept the call, byte-for-byte identical math to
   PyTorch's own implementation otherwise, and the same algorithm this project's own
   `newton_schulz_reference` — already correctness-tested — implements) gave **~21% higher
   throughput** (2,277.5 vs 1,886 tok/s) with **essentially identical quality** (val CE 4.800
   vs 4.803, PPL 121.5 vs 121.9 — within noise) at identical VRAM. Unlike `ns_steps=3`, this
   is not a tradeoff: it is a straightforward correctness-preserving speed fix for
   BF16-emulating hardware.
5. **Quality-per-wall-clock-time reverses the earlier quality-per-token conclusion — AdamW
   wins.** The original report's own "What this does not settle" section flagged this as the
   real open question: Muon's best arm beat AdamW's best arm on quality *per token*, but Muon
   also took ~2.3x longer per token, so whether Muon would still win given AdamW the same
   *wall-clock budget* (and therefore ~2.3x more tokens) was untested. It is now tested:
   AdamW lr=1e-3 run for 11,700 updates (23.94M tokens, 5,602s wall — matching Muon
   lr=3e-3/ns_steps=5's 5,423s target) scored **val CE 4.504 / PPL 90.38** — clearly better
   than every Muon arm in this report, including the best one found (lr=1e-2: PPL 110.75) and
   the fastest-clean one (fp32-ns: PPL 121.5). **At matched wall-clock time — the metric that
   actually matters for a fixed-compute-budget training run — AdamW is the better choice on
   this hardware, not Muon.** This does not contradict finding 1's earlier
   quality-per-*token* result; it answers a different, more decision-relevant question for
   MF-070's actual 350M run, which will be wall-clock-budget-limited on this single 8 GB card.
6. **A real, direct test of the best-case Muon "hybrid" (FP32 Newton-Schulz + `ns_steps=3` +
   lr=1e-2, no compile) confirms AdamW still wins — decisively, not marginally.** After finding
   5, the user asked whether combining the two real speed wins (finding 2's `ns_steps=3` and
   finding 4's FP32 Newton-Schulz) at Muon's best known LR (finding 1's 1e-2) might close the
   gap. A first-pass answer used extrapolation: fitting `time_per_step = T_fixed + ns_steps *
   cost_per_iteration` from the three real per-step-time measurements above gave `T_fixed ~=
   0.479s`, `cost_per_iteration(fp32) ~= 0.084s` (vs `~= 0.121s` for bf16 — a clean, direct
   numeric confirmation of finding 4's BF16-emulation-overhead claim), predicting the hybrid at
   `~2,801 tok/s`; applying AdamW's own real token-scaling slope (CE dropped 5.130 → 4.504,
   i.e. ~0.736 nats per e-fold of tokens, from 10.23M → 23.94M tokens) to Muon's per-token
   quality data suggested the hybrid at lr=1e-2 might reach **PPL ≈ 84**, edging out AdamW's
   90.38. At the user's request, this was tested directly rather than left as extrapolation:
   7,700 updates (~15.75M tokens), wall-clock-matched to AdamW's 5,602s. **Real result:
   2,892.3 tok/s (close to the fitted 2,801 estimate), 5,447s wall (slightly *under* AdamW's
   budget, so the comparison is not biased in AdamW's favor), val CE 4.563 / PPL 95.86 / BPB
   1.535** — worse than AdamW's real 4.504 / 90.38 by a clear ~6% relative PPL margin, and
   worse than the earlier extrapolated estimate of ~84. **The extrapolation was wrong** — it
   overestimated how much Muon's per-token quality edge would compound at a larger token
   budget; AdamW's own token-scaling turned out stronger than the borrowed-slope assumption
   captured. This resolves finding 5's residual uncertainty with a real, non-extrapolated
   result: **AdamW wins at matched wall-clock time, including against the best Muon
   configuration this project has found.** Peak VRAM: hybrid Muon 5.08 GB vs AdamW's 5.63 GB
   — the one axis where Muon still has a real, measured edge.

## What this does not settle

- **Muon's true LR ceiling remains open** (finding 1) — 1e-2 was not the top of a converged
  curve, only the top of this pass's tested range.
- **The `torch.compile`-on-Muon corruption's exact root cause is not diagnosed** (finding 3)
  — the practical recommendation (avoid it) does not depend on knowing the mechanism, but a
  deeper investigation was out of scope for a bounded pre-work pass.
- **The wall-clock-matched AdamW comparison used one AdamW LR (1e-3, the prior best) at one
  matched-time point** — it does not establish AdamW's own ceiling at this larger effective
  token budget, nor test whether a re-tuned LR for the larger budget would do even better. If
  AdamW's own LR were re-tuned for this larger budget, its real advantage over Muon (finding 6)
  could plausibly be even larger, not smaller.
- **Single seed, ~10-24M tokens per arm** — far short of the frozen 3B-token target for the
  actual 350M run. Exploratory, not scale-representative, matching every other MF-070
  pre-work pass this session.
- Other "cutting-edge" Newton-Schulz research directions discussed earlier this session
  (Gram Newton-Schulz — already effectively what `torch.optim.Muon` does internally per
  finding 4's source read, so not a separate untested idea; Hierarchical/Tiled Newton-Schulz;
  Chebyshev-type polynomial acceleration; "Turbo-Muon" spectral preconditioning) were
  deliberately not implemented here — they would require hand-rolling unverified numerical
  code for an unattended multi-hour run with no way to catch a subtle correctness bug before
  it consumed the full GPU budget, an unacceptable risk profile given finding 3 already shows
  how easily an optimizer-internals change can silently corrupt training on this hardware.

**Conclusion for MF-070's 350M run: use AdamW.** Findings 5 and 6 together settle this with
real, matched-wall-clock evidence, not extrapolation: **AdamW beats every tested Muon
configuration, including the best-case hybrid combining every real Muon speed win found in
this pass at Muon's best known LR.** This reverses the original report's "Muon wins on
quality" framing once the comparison is done on the metric that actually matters for a
fixed-compute-budget run. Muon's only remaining real advantage is lower peak VRAM (5.08 GB vs
AdamW's 5.63 GB) — worth revisiting only if MF-070's actual 350M profiling run turns out to be
VRAM-bound rather than time-bound. If Muon is used regardless, `fp32-ns-3e-3`'s configuration
(FP32 Newton-Schulz, no compile) is the correct base — never `--compile-optimizer-step`, and
expect `ns_steps=3` to trade real quality for real speed rather than being free.
