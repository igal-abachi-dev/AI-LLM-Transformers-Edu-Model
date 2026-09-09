# MF-090 follow-up — fertility against real external tokenizers (StarCoder2, SmolLM2, Mistral, Llama 2)

## Question

Every fertility comparison this project has run so far (`artifacts/mf090-tokenizer-fertility/fertility.json`)
compared *our own* tokenizer variants (16k vs 32k, digit-split variants) — all trained on the same corpus,
by the same pipeline. That answers "does raising our own vocab size help," but not "how does our tokenizer's
actual fertility compare to real, production tokenizers other real models ship with." Raised directly by the
user after external feedback recommended StarCoder/SmolLM's 49,152-token tokenizer or Mistral/Llama 2's
32,000-token format as alternatives.

## Method

Real, no-GPU fertility measurement (bytes/token), reusing the exact held-out text `scripts/
measure_tokenizer_fertility.py` already uses for MF-090's own triage: `data/shards/mf064-150m-train/
validation`, decoded back to UTF-8 via our own 16k tokenizer (real, held-out FineWeb-Edu text, never part of
any tokenizer's own training corpus) — 1,869,374 UTF-8 bytes.

Four real, public tokenizers loaded via `transformers.AutoTokenizer.from_pretrained` (network access
confirmed working; no local file modification, no training, no redistribution — a one-off measurement
against text this project already holds):

| Label | Hub repo | Real vocab size |
|---|---|---|
| SmolLM2 | `HuggingFaceTB/SmolLM2-135M` | 49,152 |
| StarCoder2 | `bigcode/starcoder2-3b` | 49,152 |
| Mistral-7B-v0.1 | `mistralai/Mistral-7B-v0.1` | 32,000 |
| Llama-2-7b | `NousResearch/Llama-2-7b-hf` (ungated mirror of `meta-llama/Llama-2-7b-hf`, which is access-gated) | 32,000 |

**Correction to the originally-cited numbers**: the external feedback said "32,768 (Mistral/Llama 2
format)." The real number, confirmed by loading both tokenizers directly, is **32,000**, not 32,768 — a
real BPE vocabulary, not a rounded power of two.

## Real result

```
                  ours (16k)  vocab= 16384  tokens=   436,227  bytes/token=4.2853
             SmolLM2 (49152)  vocab= 49152  tokens=   407,802  bytes/token=4.5840
          StarCoder2 (49152)  vocab= 49152  tokens=   474,771  bytes/token=3.9374
     Mistral-7B-v0.1 (32000)  vocab= 32000  tokens=   444,006  bytes/token=4.2102
          Llama-2-7b (32000)  vocab= 32000  tokens=   460,972  bytes/token=4.0553
```

Fertility relative to ours (negative = they are more fertility-efficient than us; positive = we are more
fertility-efficient than them):

| Tokenizer | Vocab | Bytes/token | Relative to ours |
|---|---|---|---|
| SmolLM2 | 49,152 | 4.5840 | **+6.97%** (we are more efficient) |
| StarCoder2 | 49,152 | 3.9374 | −8.12% (they are more efficient) |
| Mistral-7B-v0.1 | 32,000 | 4.2102 | −1.75% (they are more efficient) |
| Llama-2-7b | 32,000 | 4.0553 | −5.37% (they are more efficient) |

Our own 16k number (4.2853) matches `artifacts/mf090-tokenizer-fertility/fertility.json`'s already-recorded
value exactly, confirming this reused the identical held-out sample.

## What this actually shows

**Vocabulary size alone does not determine fertility.** SmolLM2 has exactly 3x our vocabulary and is
measurably *worse* on our own held-out web text (+6.97% more bytes/token than ours) — despite StarCoder2
sharing that exact same 49,152 vocabulary size and scoring meaningfully *better* (−8.12%). The gap between
SmolLM2 and StarCoder2 at an identical vocab size is bigger than the gap between either of them and our own
16k tokenizer. What actually matters is training-corpus match and merge quality, not the raw token-count
budget — a real, direct illustration of exactly the point MF-090's own fertility triage already made about
our internal 16k-vs-32k comparison (fertility and vocabulary size are correlated, not equivalent).

**Our tokenizer is competitive, not dominant, against real production tokenizers roughly 2-3x our size.**
We beat SmolLM2 outright; we lose to StarCoder2, Mistral, and Llama 2 by real but modest margins (1.75%-8.12%
worse fertility). None of these deltas are of the same order as the ~5-7% fertility gain our own 16k→32k
tokenizer got from doubling vocabulary in MF-090's own internal comparison — meaning even a real 2-3x larger
outside vocabulary does not automatically buy a proportionally larger fertility win, consistent with
diminishing returns as vocabulary grows.

**This does not reopen the vocabulary-size question MF-087/MF-088/MF-090 already settled.** Fertility is
one input to quality, not the same thing as it (this report's own methodology note, inherited from
`measure_tokenizer_fertility.py`, says so explicitly) — MF-090's real trained-model comparison already found
32k *scored worse* on held-out BPB despite *better* fertility than 16k, at this project's own realistic
token budget. A production tokenizer's fertility edge here does not predict it would win a real trained
comparison at our budget any more than our own 32k variant's fertility edge did.

## Caveats

- Single held-out sample (FineWeb-Edu web text only) — no code, no math, no multilingual text. StarCoder2's
  code-trained tokenizer in particular might show a different, likely much larger, advantage on real code
  text; not tested here.
- `meta-llama/Llama-2-7b-hf` itself is access-gated on the Hub; `NousResearch/Llama-2-7b-hf` (a commonly
  used ungated mirror hosting the identical tokenizer files) was used instead. Not independently verified
  byte-for-byte against the official gated repo beyond the matching, expected 32,000 vocab size.
- No trained-quality comparison was run or is planned against these external tokenizers — retraining this
  project's own model under someone else's tokenizer would require throwing away tied-embedding/vocabulary
  assumptions baked into every existing checkpoint, well outside this fertility-only question's scope.
