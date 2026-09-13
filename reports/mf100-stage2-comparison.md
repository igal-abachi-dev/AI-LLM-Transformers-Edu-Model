# MF-100 stage 2: real bounded GPU comparison — gpt2-style vs. gpt4-style vs. o200k pre-tokenizer regex

Real measurements via `scripts/compare_tokenizers.py`, `150m-modern.toml`, RTX 2070
Super, FP16, seed 42, wall-clock-matched at 2560s/arm (~2,560,000,000 total
budget, ~10.7-10.9M real tokens/arm at this hardware's real throughput). All
three arms use the identical frozen 16,384 vocabulary and 14-token special-token
contract, differing *only* in pre-tokenization regex, and train on the identical
FineWeb-Edu document range (`data/shards/mf100-stage2-*-train`, matching
[[MF-097]]'s own document range) so this isolates exactly one variable per the
same discipline every other bounded comparison this session has used.

## Real results

**Bits-per-byte (BPB), not raw cross-entropy, is the fair comparison metric
here** — the three tokenizers produce different token counts for the same
text, so raw CE/PPL are not directly comparable across arms (the same
principle [[MF-087]]/[[MF-090]]'s own 16k-vs-32k vocabulary comparison
already established).

| Arm | Tokens/second | Cross-entropy | Bits/byte | vs. gpt2-baseline (relative BPB) |
|---|---|---|---|---|
| gpt2-baseline (current default) | 4,161.6 | 5.1260 | 1.72444 | — |
| gpt4-style (cl100k_base regex) | 4,251.0 | 5.1541 | **1.72165** | **−0.162% (real, small win)** |
| o200k (o200k_base case-transition regex) | 4,200.1 | 5.1776 | 1.72906 | +0.268% (real loss) |

**Raw cross-entropy alone would have been misleading here**: both alternatives
show *higher* raw CE than the baseline, which would naively look like a loss
for both. But gpt4-style needs fewer tokens to cover the same real bytes
(confirmed by stage 1's fertility result, `reports/mf100-fertility-triage.md`:
+0.73% fewer tokens), so each of its tokens carries more bytes of prediction —
once corrected for that via BPB, gpt4-style is actually a **real, small
improvement**, not a loss.

## A real, informative split between the two candidates

Both gpt4-style and o200k won stage 1's fertility check (+0.73%/+0.75% fewer
tokens respectively) — but they diverge here: gpt4-style's fertility win
*carries over* to real trained-quality BPB, while o200k's does not (a real
regression instead). This mirrors, on a smaller scale, [[MF-087]]/[[MF-090]]'s
own earlier, larger finding that a 32k vocabulary's real fertility win did
*not* carry over to trained BPB — a second, independent confirmation that
fertility and trained quality are correlated, not identical, and each
candidate needs its own real trained-quality check, not just a fertility
proxy.

## Decision

**Per [[MF-100]]'s own acceptance criterion** ("if it does win, the GPT-4-style
tokenizer becomes the default for Modern only... Edu keeps the current
plain-`ByteLevel` tokenizer as its default"): gpt4-style's real, small,
corroborated BPB win (−0.162%, ~3.5x this project's own established noise
floor) crosses that threshold. **gpt4-style becomes the Modern-only default
pre-tokenizer regex.** Edu is unaffected (Edu never had a pretokenizer
variable to begin with — this was always scoped Modern-only). o200k is not
adopted (a real, measured BPB regression, ~5.8x noise floor).

## Limitations

Single seed, single scale, single bounded token budget (~10.8M tokens/arm),
as with every other bounded comparison this session. The BPB deltas here are
real but modest — a repeat at a different seed or a larger budget could
plausibly shift the exact magnitude, though the *direction* (gpt4-style wins,
o200k loses, both against fertility-alone predictions in o200k's case) is
consistent with two independent lines of evidence (fertility + BPB) for
gpt4-style specifically, which strengthens confidence in that one result more
than a single-metric win would.
