# MF-090 — periodic-validation rerun: does the 32k-vs-16k BPB gap close or widen?

## Command

```
./.venv/Scripts/python.exe scripts/compare_tokenizers.py \
  --arm "16k;configs/scratch-150m-modern-16k-reference.toml;data/tokenizer-16k-original;data/shards/mf064-150m-train/train;data/shards/mf064-150m-train/validation" \
  --arm "32k;configs/150m-modern.toml;data/tokenizer;data/shards/mf087-32k-train/train;data/shards/mf087-32k-train/validation" \
  --output artifacts/mf090-periodic-validation --seconds 1800 --validation-interval-seconds 300 \
  --seed 42 --device cuda
```

Both arms retrained fresh (a new `--output` directory, so neither reused MF-087's existing
checkpoints), seed 42, wall-clock-matched, RTX 2070 Super, 2026-09-07/08. Raw record:
`artifacts/mf090-periodic-validation/comparison.json`.

## The question this run exists to answer

MF-087's original single-endpoint result (16k BPB 1.8039, 32k BPB 1.8212, +0.956% relative,
32k worse) left open whether that gap reflects a genuine tokenizer-quality difference or a
short-token-budget artifact (32k's larger tied-embedding table averages far fewer occurrences
per vocabulary entry at this ~7-8M-token budget than at the 3B-token release target — the
Zipf-driven explanation raised by a third code-review round). If the second explanation were
right, a curve of BPB over the training budget should show the gap **narrowing** as 32k's rarer
tokens get more exposure. If it is not, the gap should stay flat or **widen**.

## Real result — the gap crosses over, then widens against 32k

| training t (s) | 16k BPB | 32k BPB | gap (32k − 16k) | gap % | 16k CE | 32k CE | CE gap % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| ~307 | 2.1712 | 2.0895 | **−0.0817** | **−3.76%** | 6.4540 | 6.5461 | +1.43% |
| ~615 | 2.0419 | 1.9893 | **−0.0526** | **−2.58%** | 6.0696 | 6.2322 | +2.68% |
| ~922 | 1.9441 | 1.9263 | **−0.0178** | **−0.92%** | 5.7789 | 6.0348 | +4.43% |
| ~1231 | 1.8788 | 1.8813 | +0.0025 | +0.13% | 5.5850 | 5.8941 | +5.53% |
| ~1539 | 1.8366 | 1.8386 | +0.0020 | +0.11% | 5.4595 | 5.7601 | +5.51% |
| ~1802 (final) | 1.8016 | 1.8150 | **+0.0134** | **+0.74%** | 5.3555 | 5.6863 | +6.18% |

(Negative gap = 32k ahead; positive = 16k ahead. Percentages are relative to the 16k value at
that point.)

**32k starts ahead on BPB** — at the earliest checkpoint (~5 minutes in), 32k's BPB is 3.76%
*better* than 16k's, consistent with its real fertility advantage (MF-090's own fertility
triage: 32k is +5.4% more fertility-efficient than 16k on real held-out text). **Then it
crosses over around the 1,200-second mark**, and from there **16k's lead grows for the rest of
the budget** (+0.13% → +0.11% → +0.74%), ending at a magnitude close to MF-087's original
single-endpoint delta (+0.956%). This run's own final gap, +0.743%, independently reproduces
MF-087's finding in both direction and rough magnitude — a fresh retrain, different exact
checkpoint, same qualitative and near-identical quantitative result.

**This does not look like the short-budget-artifact recovery pattern.** If 32k's disadvantage
were simply "the larger embedding table needs more exposure and would close the gap given more
of the same kind of training," the curve should show 32k *narrowing* the gap as training
proceeds — instead it shows the opposite: 32k's early lead erodes, crosses zero, and then 16k's
lead widens through the rest of the tested window. Cross-entropy tells the same story more
starkly: the CE gap (32k minus 16k, both relative to 16k) grows monotonically throughout,
from +1.43% at the first checkpoint to +6.18% at the end — CE is not converging, it is
diverging, within this bounded budget.

## What this does and does not settle

**Settles, within the tested budget:** the "32k just needs more of the same tokens to close
the gap" hypothesis is not supported by this bounded-budget curve — the gap moves in the
opposite direction. Combined with the fertility triage (32k is *more* fertility-efficient, not
less, so raw tokenization inefficiency doesn't explain the regression either), the most
defensible reading of the evidence so far is that, at this project's ~150M-parameter scale and
within a budget on the order of ~10M tokens, a 32k tied-embedding vocabulary is a real, measured
quality cost relative to 16k — not an artifact of digit-splitting (the no-digit-split fertility
number rules that out as the primary cause) and not (within this window) something that
resolves itself simply by continuing to train on the same schedule.

**Does not settle:** whether this reverses at genuinely larger scale. Tao et al.'s
compute-optimal vocabulary-scaling argument (the original basis for the 32k decision, per
`docs/IMPLEMENTATION_DECISIONS.md`) was calibrated for training to the *compute-optimal* token
count at this project's ~138M non-vocab parameter scale — roughly the frozen 3B-token release
target. This test covers ~7.5-7.8M tokens, **~0.26% of that target** — even a genuine, real
crossover requiring a much larger budget than "a bit more of the same 10M-token regime" (e.g.
hundreds of millions of tokens, not billions) would be invisible to this test. This run rules
out the *simplest* version of the recovery story (linear, fast catch-up within the same
bounded-budget regime); it cannot rule out a real but slower-arriving crossover at meaningfully
larger scale.

**A secondary, incidental finding worth recording:** this run retrained the 16k arm a third
time overall (MF-087's original seed-42 run, MF-088's seed-43 run, and this seed-42 rerun),
giving three real BPB measurements of nominally the same config: 1.8039 (MF-087), 1.8048
(MF-088, seed 43), 1.8016 (this run, seed 42 again). The 3-point range (1.8016-1.8048, ~0.18%
relative) is wider than the single seed-42-vs-43 pair's own +0.046% estimate in
`reports/mf088-seed-variance.md` — expected, since CUDA training here runs with
`deterministic=False` (see `seed_everything`'s `deterministic=args.device == "cpu"` policy in
`compare_tokenizers.py`), so even the *same* seed does not reproduce bit-identical results on
CUDA; the seed-42-vs-43 pair understates true run-to-run variance by conflating it with pure
RNG-seed variance alone. This does not change the conclusion above (even ~0.18% is still ~4x
smaller than the final +0.74-0.96% tokenizer effect), but the true noise floor should be read
as "at least ~0.05%, plausibly closer to ~0.15-0.2%" rather than exactly 0.046%.

## Caveats

- Single seed pair per arm (42 for both arms here; the wider 3-point 16k range above is
  incidental, not a designed sweep) — a real replication, not a full distribution.
- Bounded ~7.5-7.8M-token budget, ~0.26% of the frozen 3B-token release target — see "does not
  settle" above.
- 6 periodic checkpoints per arm (every ~300s of training-only elapsed time) is a coarse curve;
  a real crossover's exact location (here, ~1,200s) is bracketed, not pinpointed.
- Digit-splitting's fertility cost is a separate, already-quantified effect (MF-090's fertility
  triage) and is not re-isolated by this specific run — this run's 32k arm is the shipped
  MF-087 tokenizer (digit-split, no leading space), not the no-digit-split or leading-space
  variants.
