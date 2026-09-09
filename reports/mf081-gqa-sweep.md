# MF-081: GQA-ratio bounded ablation (2 arms)

Real, matched-token, single-seed comparison of a more aggressive GQA ratio
(fewer KV heads relative to query heads) against the frozen `150m-modern`
default (`n_heads=12, n_kv_heads=4`, 3:1), following an external citation that
Muse Glimmer reportedly uses a more aggressive ratio. No new code was needed
(GQA was already fully implemented and tested, MF-039) -- this is a pure
`n_kv_heads` config sweep.

## Setup

- Config: `configs/150m-modern.toml` (baseline, n_kv_heads=4) plus
  `configs/scratch-mf081-gqa-2kv.toml` (n_kv_heads=2, 6:1) and
  `configs/scratch-mf081-gqa-1kv.toml` (n_kv_heads=1, 12:1 / MQA).
- Data/training/validation/noise-floor reference: identical to
  `reports/mf081-ln-scaling.md` (same shards, seed 42, 5,000 updates,
  batch=2, RTX 2070 Super FP16; same 435,798-token/1,868,949-byte validation
  set via `scripts/eval_checkpoint.py`).

## Results

| Arm | `n_kv_heads` | Ratio | Cross-entropy | Perplexity | Bits/byte |
|---|---|---|---|---|---|
| baseline | 4 | 3:1 | 5.154917 | 173.281 | 1.734139 |
| 2kv | 2 | 6:1 | 5.154123 | 173.144 | 1.733872 |
| 1kv | 1 | 12:1 (MQA) | 5.167437 | 175.464 | 1.738351 |

Deltas vs. baseline:

| Arm | ΔCE | ΔPPL | ΔBPB |
|---|---|---|---|
| 2kv | -0.015% | -0.079% | -0.015% |
| 1kv | +0.243% | +1.260% | +0.243% |

## Reading

- **6:1 (2kv) is indistinguishable from the frozen 3:1 default** -- the
  delta (-0.015%) sits well inside the estimated noise floor
  (~0.046-0.2%, MF-088). Halving KV heads again from the current default
  costs nothing measurable at this scale/budget.
- **12:1 / MQA (1kv) shows a small, real-looking regression** (+0.24% CE,
  +1.26% PPL) -- above the lower noise-floor estimate (0.046%) but within
  the upper estimate (~0.15-0.2%) once compounded across three metrics that
  move together, so this reads as *plausibly real but not conclusive on a
  single seed*, not a confidently-established regression.
- **Net**: no evidence supports adopting a more aggressive GQA ratio than
  the current 3:1 default at this project's scale. 6:1 is a free lateral
  move (lower KV-cache memory, same quality) if that mattered for another
  reason, but no quality upside was found to motivate the switch on its own;
  going further to MQA (1kv) trades a small, uncertain quality cost for
  KV-cache savings this project has not needed. The externally-cited Muse
  Glimmer precedent does not transfer as a quality win at this project's
  own scale/data/hardware -- consistent with how every other
  externally-suggested value this session (the `global_rope_theta`
  candidate, `head_dim=128`) has been treated: a real prior worth testing,
  not evidence on its own.
- `n_kv_heads` stays at 4 (3:1) in `150m-modern.toml`; no preset change.

## Limitations

Same as `reports/mf081-ln-scaling.md`: single seed, single ~10.24M-token
bounded budget, held-out validation quality only.
