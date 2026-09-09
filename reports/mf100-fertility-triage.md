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
comparison this session) is now warranted per the task's own rule. **Not
run in this pass** -- queued, pending a decision on where it slots into
the current real-GPU-arm queue (MF-101/083/095/097 already ahead of it).
If stage 2 confirms a real trained-quality win too, the task's own
acceptance criterion already specifies the outcome to adopt: Modern-only
by default if it wins, Edu keeps the current regex either way.
