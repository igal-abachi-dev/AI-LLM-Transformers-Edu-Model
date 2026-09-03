# MF-070 pre-work: RoPE-vs-NoPE long-context extrapolation — RTX 2070 Super

Environment: same as `reports/mf070-pre-muon-vs-adamw.md`. This is a follow-up to
`reports/mf070-global-nope-quality.md`, addressing the question that test's own scope
explicitly cannot answer: does NoPE actually help **length generalization** (quality at
sequence lengths beyond training length), as NoPE research claims, rather than just
in-distribution quality at the trained length?

## Method

Both `nope-rope` and `nope-none` checkpoints (from the quality test above) were trained
on `sequence_length=1024` packing. A fresh, real, held-out FineWeb-Edu document pool
(pinned revision, documents 327,000+, past every range used anywhere else this session)
was packed at `sequence_length=2048` — the model's `max_seq_len`, never seen at that
length during training — via `scripts/prepare_data.py`: 4,370,432 non-padding tokens
across 2,134 sequences. A scratch driver
(`eval_long_context_extrapolation.py`, not tracked) computes per-position cross-entropy
(unreduced `F.cross_entropy`, matching `evaluate_token_batches`'s own pattern) and buckets
it by absolute position: **early** (positions 0–1023, in-distribution) vs **late**
(positions 1024–2047, genuinely beyond training length — the actual extrapolation
regime). Raw records: `artifacts/mf070-nope/nope-*/long-context-extrapolation.json`.

## Results

| Checkpoint | Early CE | Early PPL | Late CE | Late PPL | Late − Early CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| rope | 5.138 | 170.4 | 5.126 | 168.4 | **−0.0119** |
| none (NoPE) | 5.150 | 172.4 | 5.131 | 169.2 | **−0.0186** |

(Negative Δ means the "late," beyond-training-length segment scored *better* than the
"early," in-distribution segment — i.e. no degradation from extrapolating to 2x training
length, for either checkpoint.)

## Findings

1. **Neither checkpoint degrades extrapolating to 2x the trained sequence length.** Both
   actually score slightly *better* on the late segment than the early one — plausibly
   because more accumulated context simply makes next-token prediction easier in general,
   an effect independent of position-encoding scheme, at least over this modest 2x range.
2. **RoPE stays ahead of NoPE in absolute terms at both segments** (170.4 vs 172.4 early;
   168.4 vs 169.2 late) — consistent with the in-distribution quality test.
3. **NoPE's improvement from early→late is marginally larger than RoPE's** (−0.0186 vs
   −0.0119 nats) — but the difference between the two deltas is only ~0.007 nats, well
   within single-seed noise at this scale. Not a real, defensible extrapolation advantage
   for NoPE based on this data.

## Honest interpretation

**This test does not show NoPE providing a measurable extrapolation advantage over RoPE**
at this scale (150M parameters) and this extrapolation range (2x, 1024→2048 tokens).
RoPE's relative-position design is already known to generalize reasonably well over
modest length increases — which is consistent with what's observed here for *both*
encodings, not a NoPE-specific effect. NoPE's claimed benefits in the literature typically
concern much larger scale and/or much longer extrapolation ranges than tested here; this
result should not be read as "NoPE doesn't work," only as "this specific bounded,
single-seed, 2x-range, 150M-parameter test found no advantage."

## What this does not settle

- Single seed, one checkpoint per arm, only a 2x extrapolation range (1024→2048) — NoPE's
  literature claims often involve much larger multiples and/or much larger models.
- No test beyond `max_seq_len=2048` was possible (the model's configured architectural
  limit for both checkpoints).
- `local_window=512` is fixed at less than either segment length in every arm — local
  layers see the same relative window regardless of absolute position; only the one
  global layer's positional treatment (rope vs none) is under test.

**Conclusion: no evidence from this pass supports switching to NoPE for extrapolation
benefits. RoPE remains the correct, evidence-backed choice for MF-070's 350M run.**
