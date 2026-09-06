# MF-087: 16k vs 32k tokenizer, wall-clock-matched real result — and a confound

## Command

```
./.venv/Scripts/python.exe scripts/compare_tokenizers.py \
  --old-config configs/scratch-150m-modern-16k-reference.toml --old-tokenizer data/tokenizer-16k-original \
  --old-train-shards data/shards/mf064-150m-train/train --old-validation-shards data/shards/mf064-150m-train/validation \
  --new-config configs/150m-modern.toml --new-tokenizer data/tokenizer \
  --new-train-shards data/shards/mf087-32k-train/train --new-validation-shards data/shards/mf087-32k-train/validation \
  --output artifacts/mf087-tokenizer-quality --seconds 1800 --device cuda
```

Two prior attempts crashed before this one completed: a CUDA OOM during validation (fixed —
free gradients/optimizer leftovers and shrink the validation batch before allocating validation
tensors) and a FlexAttention block-mask-cache bug (MF-089, fixed and regression-tested). The 16k
arm's checkpoint from the first crashed attempt was reused rather than retrained (verified
identical admission/document range); its `wall_seconds` below is therefore the `--seconds`
budget it was run with, not a value this final invocation measured itself.

## Real result (`artifacts/mf087-tokenizer-quality/comparison.json`)

| Metric | 16k (old) | 32k (new, +digit-split) | Δ (relative) |
|---|---|---|---|
| Tokens consumed (1800s budget) | 7,570,200 | 7,109,850 | −6.1% |
| Tokens/second | 4,205.7 | 3,943.7 | −6.2% |
| Train loss | 5.134 | 5.649 | worse |
| Validation cross-entropy | 5.362 | 5.706 | worse |
| Validation perplexity | 213.2 | 300.5 | worse |
| **Validation bits/byte** | **1.8039** | **1.8212** | **+0.96% (worse)** |

BPB is the metric this comparison exists to answer, since it is tokenizer-independent by
construction (unlike CE/PPL, which are not comparable across different vocabularies). At matched
wall-clock time, the 32k+digit-split tokenizer is measurably worse than the 16k baseline on this
bounded (~7-7.5M token) run. The tokens/second drop (−6.2%) is in the same direction and rough
magnitude as the FLOPs-based prediction recorded before this run (~+9% forward compute at 32k
vocab, so roughly −8% tokens/second at fixed wall-clock) — that part of the prediction held up.

## Why this is not yet a decision — the confound

This comparison changes **two variables at once**, and they push bytes/token in opposite
directions:

- **Vocabulary size** (16k → 32k) should *lower* fertility (fewer tokens for the same text).
- **Digit-splitting** (added in the same tokenizer retrain) *raises* fertility — verified
  empirically before this run: the current pattern (`\d{1,3}`, no leading-space allowance)
  pre-tokenizes `" 2026"` as `['Ġ', '202', '6']` (a wasted lone-space token), while a
  leading-space-allowing variant (`" ?\d{1,3}"`) produces `['Ġ202', '6']`.

The real result above cannot distinguish "the larger vocabulary doesn't pay for its own compute
cost at this scale" from "digit-splitting's fertility cost is eating the vocabulary's fertility
gain" from "both" from "single-seed noise at a bounded token budget far short of the 3B-token
release target." See MF-090 (`tasks/backlog.md`) for the follow-up needed to actually answer
that: a third arm (32k, no digit-split) and/or the leading-space digit-split variant, before any
final tokenizer decision is made.

## Limitations

- Single seed (42), single wall-clock budget (1800s/arm) — not a sweep.
- The two arms train different-vocabulary-size models (different embedding table shapes), not a
  matched-parameter comparison.
- ~7-7.5M tokens/arm is far short of the frozen 3B-token release target; a real difference at
  this bounded scale is not guaranteed to hold (or to have the same sign) at 3B tokens.
- The 16k arm's `wall_seconds` (1800.0) is the target budget, not an independently re-measured
  value, since that checkpoint was reused rather than retrained (see Command section above).
