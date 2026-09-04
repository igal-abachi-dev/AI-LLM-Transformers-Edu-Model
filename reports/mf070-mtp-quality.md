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

## Real result

*(Not yet run. To be filled in after `scripts/compare_mtp.py` completes, with real command
output/checkpoint records per this project's own evidence discipline — never reported without
them.)*
