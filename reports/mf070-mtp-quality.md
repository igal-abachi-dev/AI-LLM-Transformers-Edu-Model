# MTP pre-work: literature-based expectation, written before running the real test

This section is written and committed *before* `scripts/compare_mtp.py` has been run for real.
The point is to make a falsifiable prediction first, so the real result below (once it lands)
can be checked against it honestly rather than rationalized after the fact.

## What MTP is, in this project's implementation

`src/minifrontier/mtp.py`'s `MTPHeads`: extra, untied linear heads reading the same final
hidden state the main `lm_head` reads, each grading a further-ahead token (`t+2`, `t+3`, ...)
via the same cross-entropy machinery as the main next-token loss, summed with a configurable
weight (see `docs/IMPLEMENTATION_DECISIONS.md`, 2026-09-04, and `AGENTS.md`'s MTP carve-out).
This is a deliberately simplified variant of the technique — one linear projection per extra
head, not a separate small transformer block per depth (DeepSeek-V3's full design).

## Literature this expectation is grounded in

- **DeepSeek-V3** (Liu et al., arXiv:2412.19437) reports MTP "consistently improves the model
  performance on most evaluation benchmarks," at 671B total / 37B active parameters and 14.8T
  training tokens — a real, positive, production result, but at a scale roughly 4,800x this
  project's parameter count and roughly 1,400x its frozen 3B-token ceiling (and vastly more
  relative to this bounded test's ~10M-token budget). DeepSeek-V3 also reports the MTP module,
  when *kept* at inference for speculative decoding, gets a ~85-90% second-token acceptance
  rate — a separate benefit this project's design does not target (heads are training-only,
  discarded after training; see `mtp.py`'s own docstring for why).
- **Gloeckle et al.**, "Better & Faster Large Language Models via Multi-token Prediction"
  (arXiv:2404.19737) — the paper establishing multi-token prediction as a training-time
  auxiliary objective, tested across a range of model sizes up to 13B parameters. Their own
  reported finding is directional but real: the benefit **grows with model size**, and is
  markedly weaker, sometimes absent or mixed, at their smaller tested scales, with the
  clearest, most consistent gains showing up on generative/coding benchmarks rather than
  general next-token perplexity. This project's 138M-parameter model sits at or below the
  smallest end of that paper's own explored range — genuinely outside where the technique has
  clear, established support, not just a small extrapolation from it.

## Prediction (recorded before the real run)

Given both sources agree the benefit scales with model size, and this bounded test runs a
138M-parameter model on only ~10M tokens (roughly 1,400x below the frozen 3B-token release
target, and far below either paper's own scale): **expect a small, uncertain effect — plausibly
a modest improvement, plausibly a wash, plausibly a slight regression — not a clear, confident
win.** This is explicitly *not* a prediction that MTP will fail; it is a calibrated
expectation that the literature does not confidently support a win at this specific scale, so
the real test is genuinely informative rather than a formality. Any of "helps," "no measurable
difference," or "slightly hurts" would all be consistent with this prediction; only a large,
unambiguous improvement (comparable in size to what larger-scale results report) would be a
real surprise relative to it.

## Design decisions recorded before the real run (2026-09-04)

Two more design questions came up after the prediction above was written, once it was clear
DeepSeek-V3's real MTP module is a small sequential transformer block (its own causal
self-attention, chained off the previous depth's hidden state plus the true token embedding
at that position) rather than an independent linear head. Recording the decisions here,
before the real result exists, for the same reason the prediction above is pre-registered.

**Module depth: keep the linear head, do not build a chained transformer block.**
DeepSeek's design bundles two separable ideas: (a) chaining/teacher-forcing — each depth
sees the true token embedding at that position, not just the shared hidden state, which
mainly matters *across multiple* depths — and (b) a full transformer block with its own
attention. This project runs `n_extra_heads=1` (t+2 only, matching DeepSeek-V3's own real
choice at 671B scale), so idea (a)'s cross-depth chaining is structurally almost moot; the
one part that would still apply at n=1 is teacher-forced conditioning on the true t+1
embedding, which is cheap (concat + linear) and recorded as a considered, not-yet-built
follow-up (see `src/minifrontier/mtp.py`'s docstring). The full chained block is a materially
bigger, riskier addition whose marginal value at n=1 neither DeepSeek-V3 nor Gloeckle et al.
isolate from teacher-forcing alone or from simply having an auxiliary loss at all — out of
scope for this bounded experiment, and not added before a scheduled, previously-crashed-once
real run.

**Loss weight: fixed at 0.3, not decayed.** DeepSeek-V3 decays 0.3 → 0.1 over 14.8T training
tokens — a schedule calibrated for a run roughly 1,400,000x longer than this bounded ~10.23M-
token test. A decay schedule has essentially no room to matter at this scale and would add a
new hyperparameter to a test whose point is answering a simpler question first. Revisit decay
only if MTP proves valuable at the project's real 3B-token release scale.

## Real result (2026-09-06)

Command actually run, in the user's own foreground terminal (after one earlier attempt crashed
on the cross-device bug fixed in `training.py`; this run redid the baseline arm from scratch
since the script has no per-arm resume logic):

```
./.venv/Scripts/python.exe scripts/compare_mtp.py --config configs/150m-modern.toml \
  --train-shards data/shards/mf064-150m-train/train \
  --validation-shards data/shards/mf064-150m-train/validation \
  --output artifacts/mf070-mtp-quality --updates 5000 --batch-size 2 --seed 42 \
  --learning-rate 3e-4 --mtp-extra-heads 1 --mtp-loss-weight 0.3 --device cuda
```

Real output (`artifacts/mf070-mtp-quality/comparison.json`), both arms trained on the exact
same 10,230,000 tokens (5,000 updates, batch=2), same seed, same starting weights:

| Metric | Baseline | MTP (n=1, w=0.3) | Δ (relative) |
|---|---|---|---|
| Train loss (primary next-token only, uncontaminated by the auxiliary term) | 5.224824 | 5.227843 | +0.058% (slightly worse) |
| Validation cross-entropy | 5.137413 | 5.129997 | −0.144% (better) |
| Validation perplexity | 170.275 | 169.017 | −0.739% (better) |
| Validation bits/byte | 1.728251 | 1.725756 | −0.144% (better) |
| Tokens/second | 4188.7 | 3977.3 | −5.05% (slower) |
| Wall time | 2442.3s (40.7 min) | 2572.1s (42.9 min) | +5.31% (slower) |

The baseline arm's numbers are bit-for-bit identical to the original (crashed-mid-run) attempt's
baseline arm, confirming full reproducibility of this bounded-comparison methodology.

### Reading the result against the pre-registered prediction

The prediction above was: "a small, uncertain effect — plausibly a modest improvement,
plausibly a wash, plausibly a slight regression — not a clear, confident win." The real result
lands squarely inside that range: a **small, consistent improvement on all three held-out
validation metrics** (CE, PPL, BPB all move the same direction, as they must — BPB is CE
rescaled by a fixed bytes/token constant, so this is an internal consistency check, not three
independent confirmations), alongside a **very slightly worse primary training loss** and a
**real, measurable wall-clock cost** (~5.3% slower). Not a large, unambiguous win of the kind
DeepSeek-V3 reports at 671B/14.8T scale — consistent with Gloeckle et al.'s finding that the
effect is weaker at small scale. Also not a wash or a regression. The result is genuinely
informative rather than a formality, exactly as the prediction anticipated.

The train-loss-slightly-worse / validation-slightly-better split is a plausible, coherent
signature of MTP acting as a regularizer (trading a sliver of primary-objective train-set fit
for better generalization) rather than noise in opposite directions — but this is a **single
seed, single configuration** (already listed as a limitation in `comparison.json` itself), so
it should be read as suggestive, not confirmed. The effect size (~0.74% relative PPL
improvement) is larger than this project's own established single-seed noise floor from the
corrected local-window test (~0.04% relative CE difference, called "no measurable difference"
there), so this is more likely a real small effect than pure noise, but a second seed would be
needed to be confident.

### Decision for MF-070

**Do not enable MTP for MF-070's 350M profiling run.** Reasoning: MTP is off-by-default per
its `AGENTS.md` carve-out, so this is a "stay with the default" decision, not a reversal.
The measured trade — a ~0.74% relative validation-quality improvement at a ~5.3% wall-clock
cost — mirrors the exact shape of the Muon-vs-AdamW decision earlier in this pre-work: a real
per-token quality edge that loses once wall-clock time (this project's actual bottleneck,
not parameter count) is weighed in. A single-seed, ~10M-token bounded result is also too
thin to justify carrying a training-time complexity/cost addition into a real multi-hour
350M run. MTP remains a documented, working, tested feature (`src/minifrontier/mtp.py`,
`scripts/compare_mtp.py`) that can be revisited with a second seed or at larger scale
(where Gloeckle et al.'s own finding predicts the effect should grow) as independent
follow-up work — it is not a blocker for MF-070 either way.

### MF-088 confidence caveat (added 2026-09-07)

The "~0.04% relative CE" noise reference cited above was never an actual seed-variance
measurement — it was the delta between two *different local_window configs* at one seed each
(see `reports/mf070-local-window-corrected.md`'s own MF-088 caveat), reused informally as a
proxy. `reports/mf088-seed-variance.md` now provides a real same-config, different-seed
measurement: **+0.046% relative CE/BPB** — coincidentally almost identical in magnitude to the
informal number, which is a coincidence, not something that should have been trusted at the
time. Against this real noise floor, MTP's own effect (CE 5.137 → 5.130, **0.136% relative,
≈3.0x the measured noise**) holds up as more likely a real small effect than pure noise, same
conclusion as before, now on firmer footing. The recommendation (MTP stays off for MF-070) is
unchanged either way, since it was already driven primarily by wall-clock cost, not by doubt
about whether the quality effect was real.
