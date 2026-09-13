# MF-100 stage 2 follow-up: gpt4-style regex, with vs. without digit-splitting

Real measurements via `scripts/compare_tokenizers.py`, `150m-modern.toml`, RTX 2070
Super, FP16, seed 42, wall-clock-matched at 2560s/arm. All four arms use the
already-adopted gpt4-style (cl100k_base) pre-tokenizer regex and the identical
frozen 16,384 vocabulary/14-token special-token contract, differing *only* in
whether and how digits get further split after the regex pass, training on the
identical FineWeb-Edu document range as every other MF-100/MF-097 arm. This
answers the question left open when gpt4-style was adopted (MF-100 stage 2,
`reports/mf100-stage2-comparison.md`): does stacking digit-splitting on top of
the now-default regex help further, given fertility alone can't say (BPB is the
fair metric, as established throughout this investigation)?

## Real results

| Arm | Tokens/second | Cross-entropy | Bits/byte | vs. gpt4-baseline (relative BPB) |
|---|---|---|---|---|
| gpt4-baseline (no digit split, current default) | 4,129.1 | 5.1752 | **1.72869** | — |
| gpt4-nolead (grouped digit split, no leading space) | 3,878.0 | 5.2020 | 1.73763 | +0.517% (real loss) |
| gpt4-leadspace (grouped digit split, leading space) | 3,868.1 | 5.1993 | 1.73674 | +0.466% (real loss) |
| gpt4-individual (`Digits(individual_digits=True)`) | 3,603.7 | 5.1756 | 1.75793 | **+1.691% (real, largest loss)** |

## Decision

**No digit-split mode beats the no-digit-split gpt4-style baseline. The
already-adopted default (gpt4-style regex, `digit_split="none"`) is confirmed,
unchanged.** All three deltas are real losses, well above this project's own
established single-config noise floor (~0.046-0.2%, `reports/mf088-seed-variance.md`):
nolead and leadspace are both ~2.3-2.6x the floor, and individual-digit-splitting
is ~8.5-37x the floor — clearly the worst of the three, consistent with (not
contradicted by) stage 1's own fertility-triage finding that individual-digit
splitting was already the real fertility **cost** among the four original
candidates (`reports/mf100-fertility-triage.md`, +2.88% more tokens vs. the
gpt2 baseline). Digit-splitting was evaluated on its own (fertility) merits
again here, at real trained quality this time, and lost again — this closes
the tokenizer regex/digit-split question `AGENTS.md`'s frozen-scope section
already documents as reverted-and-not-recombined.

## Limitations

Single seed, single scale, single ~10.2-10.6M-token bounded budget per arm, as
with every other bounded comparison this session. `predicted_tokens` differs
slightly for the individual-digit arm (439,890 vs. 432,729 for the other
three) since digit-splitting changes tokenization density even on the same
held-out text — exactly why BPB, not raw cross-entropy, is the comparison
metric used here.
