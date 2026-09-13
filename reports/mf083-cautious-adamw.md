# MF-083: real bounded comparison — plain AdamW vs. Cautious AdamW (C-AdamW)

Real measurements of `train/pretrain.py --optimizer {adamw,cautious_adamw}` on
`150m-modern.toml`, RTX 2070 Super, FP16, real data
(`data/shards/mf064-150m-train`, seed 42), 5000 updates / 10,230,000 train
tokens per arm. Both arms use identical config, data, seed, and everything
else except the optimizer. Cautious AdamW is the real published mechanism
(Liang, Chen, Liu, Liu, "Cautious Optimizers: Improving Training with One
Line of Code", arXiv:2411.16085, ICLR 2026) implemented in
`src/minifrontier/cautious_adamw.py` — masks the update to elements agreeing
in sign with the current gradient, rescaling the effective learning rate to
compensate.

## Real results

| Arm | Cross-entropy | Perplexity | Bits/byte | Tokens/second | Peak VRAM (MB) |
|---|---|---|---|---|---|
| baseline (AdamW) | 5.100674 | 164.13 | 1.71589 | 3,842.3 | 5,158.9 |
| cautious (C-AdamW) | 5.185800 | 178.72 | 1.74453 | 3,541.0 | 5,155.8 |

Relative to baseline: **+1.669% cross-entropy, +8.89% perplexity, +1.669%
bits/byte — all worse.** This is a real, decisive effect: 36.3x this
project's own measured seed-to-seed noise floor (MF-088, ~0.046% relative
CE), not a result that could plausibly be run-to-run noise.

Cautious AdamW is also **~7.8% slower** (3,541.0 vs 3,842.3 tok/s) at
essentially identical VRAM — the masking/rescaling computation adds real
per-step overhead with no offsetting speed benefit.

## Reading

**Cautious AdamW is a real, clean loss on both quality and speed at this
project's scale, data, and hardware.** The paper's own real reported result
(up to 1.47x speedup on Llama pretraining) does not transfer here — plausible
explanations, none independently confirmed by this pass: the paper's gains
are reported at a very different scale/architecture, the masking's
sign-agreement heuristic may interact differently with this project's own
architecture choices (GQA, gated attention, LayerNorm scaling), or the
effective-learning-rate rescaling (`dim / (aligned_count + xi)`) may behave
differently at this project's much smaller per-layer widths than at the
scale the paper measured.

## Decision

**Cautious AdamW is not adopted.** Plain AdamW remains the default optimizer
for [[MF-070]] and every other real training run. The implementation stays
in the codebase (tested, documented, off by default via
`TrainingConfig.optimizer = "adamw"`) as a real, available, but
not-recommended experiment — matching this project's own precedent for
negative results (e.g. Muon, MF-057) staying implemented rather than deleted.

## Limitations

Single seed, single scale (150M-modern), single token budget (~10.23M
tokens). A different scale, longer real budget, or different architecture
could in principle behave differently — but the effect measured here is
large and consistent (both quality and speed move the same, unfavorable
direction), giving no positive signal worth chasing at larger scale without
new evidence.
