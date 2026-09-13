# MF-100: GPT-4-style (cl100k_base) regex pre-tokenizer -- stage 1, real fertility triage

Real, no-GPU fertility (bytes/token) comparison between this project's
current GPT-2-style pre-tokenization regex and the real cl100k_base
(GPT-4/GPT-3.5-turbo) regex, following the same two-stage discipline MF-090
established: a cheap fertility check first, a real GPU quality comparison
only if fertility looks neutral-or-better.

## Implementation

`train_byte_bpe` gains `pretokenizer: Literal["gpt2", "gpt4"] = "gpt2"`
(frozen default unchanged). `"gpt4"` uses the real cl100k_base pattern,
verified directly against tiktoken's own primary source
(`openai/tiktoken`'s `tiktoken_ext/openai_public.py`), not a secondary
summary -- an earlier `WebSearch` summary of this exact pattern misquoted
the digit clause as `\p{N}{2,}`; checking the primary source caught it.
Ported from PCRE's possessive quantifiers to plain greedy equivalents,
since the `tokenizers` library's own `Regex` silently mis-parses the
possessive suffix as an unrelated repeated-group quantifier rather than
rejecting it -- verified directly: with the possessive form, "123456789"
pre-tokenized as one unsplit piece instead of correctly capping at 3
digits. `"gpt4"` requires `digit_split="none"` (rejected otherwise): the
GPT-4 regex already caps digit runs itself, and stacking the project's own
separate digit-isolation rule on top was never tested and has no clear
semantics.

**A real, previously-untested backward-compatibility bug was found and
fixed while building the real comparison tokenizers.** `data/tokenizer`
(the current production tokenizer, trained before MF-103 added `<|eot|>`/
`<|file_sep|>`/`<|repo_name|>`) failed to load at all once `SPECIAL_TOKENS`
grew to 14 entries: `_validate_special_tokens` required *every* token in
the current contract to be present, so an 11-token tokenizer missing the
3 newest ones was rejected outright -- which would have made every
already-published release's own tokenizer permanently unloadable the
moment this kind of change ships. Fixed: a token absent from the loaded
tokenizer is now tolerated (it predates that token's introduction, and the
model trained against it never learned anything about it either -- not
drift); a token that *is* present at the wrong ID is still rejected
unconditionally, exactly as before. 2 new regression tests
(`test_tokenizer_missing_a_newer_special_token_still_loads`,
`test_tokenizer_still_rejects_a_special_token_present_at_the_wrong_id`).

## Real fertility result

Two tokenizers trained on the exact same corpus every tokenizer this
project has trained on (`data/tokenizer-corpus.jsonl`), same 16,384
vocabulary, same 14-token special-token contract -- differing *only* in
pre-tokenization regex, isolating exactly one variable:

| Tokenizer | Tokens (held-out sample) | Bytes/token |
|---|---|---|
| `gpt2-14tok` (current default regex) | 436,231 | 4.2853 |
| `gpt4-14tok` (cl100k_base regex) | 433,036 | **4.3169** |

Held-out sample: 1,869,374 real UTF-8 bytes, decoded from
`data/shards/mf064-150m-train/validation` via `data/tokenizer` (the same
real held-out FineWeb-Edu validation text every other tokenizer comparison
this session has used, never part of any tokenizer's own training corpus).

**Real, measurable improvement: cl100k_base's regex is ~0.74% more
fertility-efficient** (fewer tokens for the same real text) than this
project's current GPT-2-style regex, at the same vocabulary size and
special-token contract.

## Per this task's own acceptance criterion: this crosses the threshold

"Only if fertility is neutral-or-better... does a real trained-quality
comparison follow." A +0.74% fertility improvement is a real positive
result, not merely neutral -- stage 2 (a real bounded matched-token GPU
training comparison, same discipline as every other `reports/mf0*`
comparison this session) is now warranted per the task's own rule.

## Stage 1, round 2 (2026-09-11): the two candidates added by the wider
survey (individual-digit splitting, o200k_base case-transition regex)

Same discipline, same corpus (`data/tokenizer-corpus.jsonl`), same 16,384
vocabulary, same 14-token special-token contract, same held-out sample
(1,869,374 real UTF-8 bytes, `data/shards/mf064-150m-train/validation`,
decoded via `data/tokenizer`) -- all four candidates measured together in
one run (`scripts/measure_tokenizer_fertility.py`,
`reports/mf-overnight-eval/mf100-fertility-stage2-candidates.json`):

| Tokenizer | Tokens (held-out sample) | Bytes/token | vs. gpt2-14tok baseline |
|---|---|---|---|
| `gpt2-14tok` (current default regex) | 436,231 | 4.2853 | -- |
| `gpt4-14tok` (cl100k_base regex) | 433,036 | 4.3169 | +0.73% fewer tokens |
| `individual-digit-14tok` (`Digits(individual_digits=True)`, gpt2 regex) | **448,788** | **4.1654** | **+2.88% MORE tokens** |
| `o200k-14tok` (o200k_base case-transition regex) | 432,951 | 4.3177 | +0.75% fewer tokens |

**Individual-digit splitting fails this task's own fertility gate --
a real, measured cost (+2.88% more tokens), not neutral-or-better.** This
is the same direction as MF-087/090's earlier finding that grouped
(<=3-digit) splitting also cost fertility, now confirmed for the finer,
one-digit-at-a-time granularity too, and more pronounced. **Per the task's
own acceptance rule, this candidate does not advance to stage 2** -- it
stays implemented and tested (real code, real tests, `tokenizer.py`) as an
available-but-not-recommended option, exactly like the two earlier
grouped-digit-split modes, not queued for GPU time.

**o200k_base passes, marginally ahead of gpt4-style even** (+0.75% vs.
+0.73% fewer tokens -- a real result, though the two are close enough that
either could plausibly lead in a repeat with different held-out text).
Both `gpt4-14tok` and `o200k-14tok` now clear the gate.

## What this says about StarCoder2's own real fertility advantage

[[MF-090]]'s external-fertility follow-up (`reports/mf090-external-tokenizer-fertility.md`)
found StarCoder2's real tokenizer **8.12% more fertility-efficient** than
ours on the same held-out web text -- and StarCoder2's real tokenizer does
use individual-digit splitting (`Sequence([Digits(individual_digits=True),
ByteLevel(...)])`, its own primary `tokenizer.json`). Given the result
above (digit-splitting alone costs fertility, +2.88%, isolated on our own
16k vocab/corpus), **digit-splitting cannot be the source of StarCoder2's
advantage -- it is a real, measured drag against it.** The −8.12% gap has
to come from elsewhere. The same MF-090 follow-up already ruled out
"bigger vocabulary always wins" as a full explanation (SmolLM2 shares
StarCoder2's exact 49,152 vocab and is +6.97% *worse* than ours on this
same held-out sample) -- but between StarCoder2 and SmolLM2, StarCoder2 is
the one carrying a real, measured fertility handicap from digit-splitting
and still winning, which makes its remaining advantage (vocabulary size
combined with corpus/merge quality) look larger and more real, not
smaller. Net: StarCoder2's fertility edge is despite its digit-splitting
choice, not because of it -- a real, own-corpus-isolated data point this
project did not have before this pass, not an assumption.

## Stage 2 scope, resolved

Three real candidates now qualify for the GPU quality comparison against
the `gpt2` baseline: **gpt4-style, o200k-style**, and the previously-added
confounded/informational **StarCoder2** arm (see MF-100's own backlog
entry for that arm's caveats). Individual-digit splitting is excluded by
its own real fertility result above. Not run in this pass -- queued behind
[[MF-095]]'s broader-eval follow-up, the next real GPU arm ahead of it.
