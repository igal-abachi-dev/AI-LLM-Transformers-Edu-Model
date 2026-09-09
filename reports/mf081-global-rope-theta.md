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
  in-distribution BPB tested here is a weak proxy for that claim. A real
  test needs the long-context retrieval eval ([[MF-086]]'s
  needle-haystack harness, the same tool [[MF-082]] uses) run against this
  checkpoint at context lengths beyond `local_window=512`, not just a
  validation-split loss number. Not run in this pass; the checkpoint
  (`artifacts/mf081-global-rope-theta/500k/final`) is available for that
  follow-up.
- No preset change; `global_rope_theta` stays unset (`None`, shared 10,000)
  in `150m-modern.toml`.

## Limitations

Single seed, single ~10.24M-token bounded budget, in-distribution
short-context validation only -- the long-context retrieval question this
value actually targets remains untested.
