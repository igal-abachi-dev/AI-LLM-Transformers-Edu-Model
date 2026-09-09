# MF-082: gated attention bounded arm -- validation quality and needle-haystack re-eval

Real, matched-token, single-seed test of per-head gated attention
(`gated_attention`, Qiu et al. arXiv:2505.06708, Qwen3-Next-style sigmoid
gate on the SDPA output before `out_proj`), the fix [[MF-082]] chose for the
real, confirmed ring-cache attention-sink degradation found in
`reports/mf086-needle-haystack.json`.

## Setup

- Config: `configs/150m-modern.toml` (baseline) vs.
  `configs/scratch-mf082-gated-attention.toml` (`gated_attention=true`,
  everything else unchanged).
- Data/training/validation/noise-floor reference: identical to
  `reports/mf081-ln-scaling.md`.

## Part 1: held-out validation quality

| Arm | `gated_attention` | Cross-entropy | Perplexity | Bits/byte |
|---|---|---|---|---|
| baseline | off | 5.154917 | 173.281 | 1.734139 |
| gated | on | 5.120597 | 167.435 | 1.722594 |

Delta vs. baseline: **ΔCE -0.666%, ΔPPL -3.374%, ΔBPB -0.666%.**

This is a real improvement well above the estimated noise floor
(~0.046-0.2%, MF-088) -- the clearest quality signal of any [[MF-081]]/
[[MF-082]] arm run this pass. Consistent with gated attention's citation as
a general quality technique (not just a sink-eviction patch), matching
[[MF-082]]'s own 2026-09-08 scope note that Muse Glimmer applies gating to
every layer for exactly this reason.

## Part 2: needle-haystack retrieval re-eval (the task's actual acceptance test)

Ran `scripts/eval_needle_haystack.py --checkpoint ... --context-lengths 512
1024 2032` against both arms' `final/` checkpoints
(`reports/mf082-needle-haystack-baseline.json`,
`reports/mf082-needle-haystack-gated.json`).

| Arm | retrieval rate @512 | @1024 | @2032 |
|---|---|---|---|
| baseline | 0.0 | 0.0 | 0.0 |
| gated | 0.0 | 0.0 | 0.0 |

**Inconclusive, not a negative result.** `reports/mf086-needle-haystack.json`
(the reference degradation this task is fixing) measured `{512: 1.0, 1024:
0.4, 2032: 0.0}` against the real MF-065 **1B-token** release checkpoint --
retrieval clearly works within the local window there and collapses beyond
it. These two bounded-comparison arms were trained for only 5,000 updates
(~10.24M tokens, ~100x less), and at that budget **neither arm retrieves at
all, even at length=512, well inside the local window** -- the needle-
retrieval capability itself has not emerged yet at this token budget, for
either arm. This comparison cannot distinguish "gated attention doesn't fix
sink eviction" from "neither checkpoint has learned in-context retrieval
yet"; it is not evidence against the fix.

**A real re-test of this specific acceptance criterion needs a
substantially larger token budget** -- either reusing an existing
larger-budget checkpoint pair (none exists with `gated_attention` on) or
training `gated_attention=true` on a budget closer to MF-065's 1B tokens
before re-running the needle-haystack comparison. Not done in this pass;
flagged as the real remaining gap in [[MF-082]]'s acceptance criterion.

## Status

`gated_attention` stays off by default (`ModelConfig.gated_attention: bool =
False`). Part 1's quality result is real and positive; Part 2's actual
sink-fix verification remains open pending a larger-budget re-test.

## Limitations

Single seed, single ~10.24M-token bounded budget for both parts; Part 2's
result is inconclusive rather than negative, for the reason stated above.
