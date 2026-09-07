# MF-088 — same-config seed-to-seed variance (real result)

## Command

```
./.venv/Scripts/python.exe scripts/compare_tokenizers.py \
  --arm "16k;configs/scratch-150m-modern-16k-reference.toml;data/tokenizer-16k-original;data/shards/mf064-150m-train/train;data/shards/mf064-150m-train/validation" \
  --output artifacts/mf088-seed-variance --seconds 1800 --seed 43 --device cuda
```

Run in the user's own foreground terminal, 2026-09-07, RTX 2070 Super, `torch` per this
project's pinned environment. The seed-42 half of the comparison is not a new run: it reuses
the already-real, already-persisted MF-087 result (`artifacts/mf087-tokenizer-quality/comparison.json`,
arm `seed-42-16k`) trained under the byte-for-byte identical config, tokenizer, shards, batch
size, learning rate, and wall-clock budget — the only variable that differs is the seed.

## Real result

| seed | completed_updates | tokens | tokens/sec | wall_seconds | loss | CE | PPL | BPB |
|---|---|---|---|---|---|---|---|---|
| 42 | 3,700 | 7,570,200 | 4,205.7 | 1800.0 (target, not exactly measured — see MF-087's own caveat) | 5.134109 | 5.362389 | 213.234 | 1.803934 |
| 43 | 3,775 | 7,723,650 | 4,272.2 | 1807.9 (measured) | 5.427844 | 5.364856 | 213.760 | 1.804764 |

Relative spread (seed 43 vs seed 42):

- **Cross-entropy: +0.046%**
- **Bits-per-byte: +0.046%**
- Perplexity: +0.247% (larger only because PPL = exp(CE) amplifies a small CE delta nonlinearly)

## What this settles, and what it does not

**Settles:** whether a same-config, single-seed effect this project has measured is distinguishable
from run-to-run noise. It is. MF-087's headline 16k-vs-32k BPB delta (1.8039 → 1.8212, **+0.956%**)
is roughly **20x larger** than this measured ~0.046% seed-noise floor. The earlier informal
noise reference (~0.04% relative CE, from the corrected local-window test) turns out to have
been approximately correct in magnitude, even though it was a different-configs delta rather
than a true seed-variance measurement — a coincidence, not something to rely on again without
checking.

**Does not settle:** *why* the 0.956% MF-087 delta exists, or whether it would hold, shrink, or
reverse at the real 3B-token release budget. Two candidate explanations remain open and are not
distinguished by this measurement: (1) the digit-splitting confound already named in MF-087's
own report, and (2) the Zipf/short-token-budget effect raised by a third code-review round —
at ~217 mean occurrences per vocabulary entry (32k) vs ~462 (16k) at this ~7.1-7.6M-token
budget, a large fraction of the larger tied-embedding table may still be near-initialization,
which would inflate BPB independent of tokenizer quality and would look very different at the
3B-token target (≈91,500 vs ≈183,000 mean occurrences). The periodic-validation rerun proposed
alongside this task (see `compare_tokenizers.py --validation-interval-seconds`, MF-090's status
note) is what would distinguish these — a BPB curve still closing the gap late in the budget is
evidence for (2); a flat or widening gap argues against it.

## Caveats

- Single seed pair (42 vs 43), single config (`150m-modern` architecture at 16k vocab), single
  bounded budget (~7.1-7.7M tokens). This estimates the noise floor at this project's exploratory
  scale, not a general claim about noise at every config/budget this project has tested.
- Not (yet) re-run for the 32k config, or for any of the other bounded comparisons this session
  produced (local-window, NoPE, MTP) — this single measurement is being used as the best available
  noise-floor estimate for annotating those conclusions' confidence, per MF-088's original intent,
  not as a per-experiment recalculation.
- `tokens_per_second`/`wall_seconds` differ slightly between seeds (4,205.7 vs 4,272.2 tok/s) —
  expected run-to-run system noise (thread scheduling, allocator behavior), not a quality signal.
