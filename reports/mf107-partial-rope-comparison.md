# MF-107: real bounded comparison — partial RoPE fractions vs. full RoPE

Real measurements of `train/pretrain.py` on `150m-modern.toml`, RTX 2070
Super, FP16, cosine schedule, `data/shards/mf064-150m-train`, seed 42, 5000
updates per arm. The 100% (full RoPE) baseline reuses [[MF-097]]'s own
`artifacts/mf097-packing/ribbon` checkpoint directly rather than retraining —
identical config, data, schedule, seed, and update count, confirmed by
direct inspection of both `trainer_state.json` files before reuse.

## Real results

| Arm | Cross-entropy | Perplexity | Bits/byte | vs. 100% (relative BPB) |
|---|---|---|---|---|
| 100% (full RoPE, current default) | 5.100674 | 164.13 | 1.71589 | — |
| 75% | 5.146111 | 171.76 | 1.73118 | +0.891% worse (19.4x noise floor) |
| 50% | 5.156728 | 173.60 | 1.73475 | +1.099% worse (23.9x noise floor) |
| 25% | 5.177751 | 177.28 | 1.74182 | +1.511% worse (32.8x noise floor) |

**Every fraction is a real, consistent loss, far above [[MF-088]]'s
established noise floor (~0.046% relative CE).** The trend is clean and
monotonic: 75% (closest to full rotation) is the least bad, 25% (furthest)
is the worst -- there is no sign of an intermediate optimum in this range,
just a smooth cost that grows as rotation coverage shrinks.

## Decision

**Full RoPE (100%) stays the default for [[MF-070]].** Neither DeepSeek-V4's
own real precedent (rotating a fixed absolute dimension count) nor
Qwen3-Next's real `partial_rotary_factor: 0.25` value transfers as a win at
this project's own scale/data/budget -- real, external precedent for the
*idea* existing elsewhere does not substitute for a real, own-corpus test,
exactly the discipline this project has applied to every other
externally-suggested value this session. This question is now closed with
real, trustworthy, decisive evidence -- no further follow-up needed before
[[MF-070]].

## Limitations

Single seed, single scale, single token budget, as with every other bounded
comparison this session. Only 25%/50%/75% were tested (not a finer sweep) --
given the clean monotonic trend already observed, a finer sweep between 75%
and 100% is very unlikely to find a fraction that beats 100% outright, so
not pursued.
