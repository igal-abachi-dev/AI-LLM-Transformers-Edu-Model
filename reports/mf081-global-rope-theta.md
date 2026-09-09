# MF-081: global_rope_theta=500,000 bounded arm

Real, matched-token, single-seed test of Llama 3's real published
`rope_theta=500,000` value (vs. this project's `10,000` default) on the
hybrid preset's global layers only (local layers unchanged at 10,000), using
the split local/global RoPE theta mechanism [[MF-081]] already adopted
directly (no ablation needed for the mechanism itself -- only this concrete
value was untested).

## Setup

- Config: `configs/150m-modern.toml` (baseline, `global_rope_theta=None` ->
  10,000 shared) vs. `configs/scratch-mf081-global-rope-theta-500k.toml`
  (`global_rope_theta=500_000`, local layers unchanged).
- Data/training/validation/noise-floor reference: identical to
  `reports/mf081-ln-scaling.md`.

## Results

| Arm | `global_rope_theta` | Cross-entropy | Perplexity | Bits/byte |
|---|---|---|---|---|
| baseline | 10,000 (shared) | 5.154917 | 173.281 | 1.734139 |
| 500k | 500,000 (global only) | 5.144027 | 171.405 | 1.730476 |

Delta vs. baseline: ΔCE -0.211%, ΔPPL -1.082%, ΔBPB -0.211%.

## Reading

- A small in-distribution improvement, right at the upper end of the
  estimated noise floor (~0.046-0.2%, MF-088) -- plausibly real, not
  conclusive on a single seed, and too small to act on alone.
- **This measurement does not actually test the value's real motivation.**
  Llama 3's `rope_theta=500,000` exists specifically for long-context
  extrapolation, not in-distribution short-context quality -- the
  in-distribution BPB tested here is a weak proxy for that claim.
- No preset change; `global_rope_theta` stays unset (`None`, shared 10,000)
  in `150m-modern.toml`.

## Needle-haystack retrieval re-eval (2026-09-09)

Ran `scripts/eval_needle_haystack.py --checkpoint
artifacts/mf081-global-rope-theta/500k/final --context-lengths 512 1024
2032` (`reports/mf081-needle-haystack-global-rope-theta.json`), the same
harness and same command shape as [[MF-082]]'s gated-attention re-eval.

| Arm | retrieval rate @512 | @1024 | @2032 |
|---|---|---|---|
| baseline | 0.0 | 0.0 | 0.0 |
| global_rope_theta=500k | 0.0 | 0.0 | 0.0 |

**Inconclusive, same reason as [[MF-082]]'s gated-attention re-eval**: this
is a 5,000-update/~10.24M-token bounded checkpoint, ~100x short of
[[MF-086]]'s reference budget (the real 1B-token MF-065 release, which
retrieves at `{512: 1.0, 1024: 0.4, 2032: 0.0}`). At this token budget
neither arm has learned in-context needle retrieval at all yet, even at
length=512 inside the local window -- this comparison cannot distinguish
"a larger global theta doesn't help extrapolation" from "retrieval hasn't
emerged yet at this budget." A real test of this value's actual motivation
needs the same larger-budget retrain this project's other undertrained
needle-haystack comparisons are now blocked on.

## Limitations

Single seed, single ~10.24M-token bounded budget for both the validation
quality and needle-haystack measurements -- the long-context extrapolation
question this value actually targets remains untested at a budget where
retrieval has actually emerged.
