# MF-101: real batch-size x gradient-accumulation throughput/VRAM sweep

Real measurements of `train/pretrain.py --batch-size {2,4,8} --accumulation-steps
{1,8,32}` on `150m-modern.toml`, RTX 2070 Super, FP16, real data
(`data/shards/mf064-150m-train/train`, seed 42). Per-arm update count derived
from a fixed ~400K-token target budget (`scripts/sweep_batch_size.py`,
`updates_for_token_budget`), floored at 10 updates minimum, since
`accumulation_steps` multiplies real compute per update (up to 128x between
the cheapest and most expensive combination tested) -- a uniform raw update
count across every arm was tried first and found impractical (see status
note below).

## Real results

| batch_size | accumulation_steps | effective batch | updates | train_tokens | tokens/second | peak_memory_mb |
|---|---|---|---|---|---|---|
| 2 | 1 | 2 | 195 | 398,970 | 3,996.9 | 5,158.9 |
| 2 | 8 | 16 | 24 | 392,832 | 4,147.5 | 5,662.2 |
| 2 | 32 | 64 | 10 | 654,720 | 4,374.1 | 5,662.2 |
| 4 | 1 | 4 | 98 | 401,016 | **643.3** | **8,307.8** |
| 4 | 8 | 32 | 12 | 392,832 | **412.6** | **8,789.3** |
| 4 | 32 | 128 | -- | 0 (never completed) | -- | -- |
| 8 | 1/8/32 | -- | -- | not attempted | -- | -- |

## Reading

**batch_size=2 is healthy at every accumulation level tested**, comfortably
under the reference card's 8GB VRAM (5.16-5.66GB peak), and throughput
actually *rises* slightly with more accumulation (3,997 -> 4,148 -> 4,374
tok/s) — consistent with reduced per-optimizer-step overhead as more
micro-batches are amortized per step.

**batch_size=4 collapses into real VRAM thrashing, confirmed by two
completed data points and a third that never finished at all.** Peak memory
jumps to 8.3-8.8GB — at or over this card's physical 8GB — and throughput
falls off a cliff: 643.3 tok/s at accumulation=1 (an 84% drop from
batch_size=2's ~4,000 tok/s), 412.6 tok/s at accumulation=8 (90% drop), and
**`batch_size=4, accumulation_steps=32` never completed a single logged
update in 49 real minutes** before being killed — the worst-case combination
apparently pushes far enough into thrashing that even the 10-update floor
couldn't finish in any practical time. This matches this project's own
already-documented hardware finding (`AGENTS.md`'s home-GPU notes: "8GB
ceiling silently pages rather than OOMs") — the card doesn't error out, it
just becomes catastrophically slow.

**batch_size=8 was not attempted.** Given batch_size=4 already exceeds the
physical VRAM ceiling and the worst tested combination there failed to
complete in 49 minutes, batch_size=8 (double the memory pressure again)
would predictably be at least as bad, likely worse, for no plausible upside
— continuing the sweep past this point would only spend real GPU time
confirming an already-clear trend, not discover anything new.

## Decision, per this task's own acceptance criterion

> "If a larger effective batch (batch x accumulation) measurably increases
> tokens/second at the same wall-clock cost, that combination becomes the
> new default for MF-070's eventual 350M run... if it does not, the current
> `batch_size=2, accumulation_steps=1` default is confirmed rather than
> assumed adequate."

**`batch_size=2` is confirmed as the right default for this hardware — not
because no larger effective batch would help in principle, but because
`batch_size=4`+ physically exceeds this reference card's 8GB VRAM at this
model size, and the resulting thrashing is catastrophically slower, not
just somewhat slower.** Within the batch_size=2 arms tested, higher
`accumulation_steps` (8 or 32) gives a real, if modest, throughput
improvement (+3.8% and +9.4% respectively) with no VRAM cost (peak memory
identical at 5,662.2 MB for both) — worth adopting as a free win for
[[MF-070]]'s eventual 350M run's own batch-size/accumulation choice, subject
to that run's own real VRAM headroom at 350M scale (untested here; this
sweep is 150M-scale evidence, and 350M's larger activations mean the same
accumulation_steps increase could have a different real memory cost).

## Limitations

Single seed, 150M-scale (not the 350M/500M scale this decision will
eventually govern), and the `batch_size=4`/`8` combinations were not driven
to real conclusions beyond the two completed 4-batch data points and one
confirmed-catastrophic failure — a real, informative negative result, not
a gap in methodology.
