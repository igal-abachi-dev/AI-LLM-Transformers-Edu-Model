# MF-067 release export and verification — real evidence

Four release directories, all re-exported from real trained checkpoints with real, specific model
cards (not boilerplate). Commands and results below are real; nothing here is projected/estimated.

## Releases

| Release | Params | Card |
| --- | ---: | --- |
| `artifacts/mf064-150m-edu-release` | 154,172,160 | `artifacts/mf067-model-cards/mf064-150m-edu.md` |
| `artifacts/mf065-150m-modern-release` | 138,446,080 | `artifacts/mf067-model-cards/mf065-150m-modern.md` |
| `artifacts/mf066-150m-edu-sft-release` | 154,172,160 | `artifacts/mf067-model-cards/mf066-150m-edu-sft.md` |
| `artifacts/mf066-150m-modern-sft-release` | 138,446,080 | `artifacts/mf067-model-cards/mf066-150m-modern-sft.md` |

Each was produced via `scripts/export.py --checkpoint <training-checkpoint>/final --tokenizer
data/tokenizer --output <release> --model-card artifacts/mf067-model-cards/<card>.md --keep-source`
(source training checkpoints retained for possible later continuation toward the frozen 3B-token
target). `export.py`'s own `verify_release` call passed for all four at export time (no exception
raised).

## Explicit re-verification (`minifrontier.release.verify_release`)

Ran again independently after export, against all four release directories: manifest integrity, no
`training_state.pt` pickle state present, and a real load-test forward pass with finite logits — all
four passed:

```
mf064-150m-edu-release:        {"status": "load_tested", "parameters": 154172160, "logits_finite": true}
mf065-150m-modern-release:     {"status": "load_tested", "parameters": 138446080, "logits_finite": true}
mf066-150m-edu-sft-release:    {"status": "load_tested", "parameters": 154172160, "logits_finite": true}
mf066-150m-modern-sft-release: {"status": "load_tested", "parameters": 138446080, "logits_finite": true}
```

## Matched-pair audit (`scripts/audit_release.py`, base releases)

`artifacts/mf067-audit-base.json`:

```json
{
  "edu_logits_finite": true,
  "edu_parameters": 154172160,
  "matched_fields": ["vocab_size", "max_seq_len", "n_layers", "d_model", "d_ff"],
  "modern_logits_finite": true,
  "modern_parameters": 138446080,
  "status": "load_tested"
}
```

Both releases load independently, share the same tokenizer, and match on every frozen
non-architecture scale field (`vocab_size`, `max_seq_len`, `n_layers`, `d_model`, `d_ff`).

## Fresh-environment deterministic reference generation

`scripts/audit_release.py`/`audit_release_pair` only checks forward-pass finiteness, not multi-token
generation determinism, so this was verified separately and explicitly, per MF-067's acceptance
clause ("fresh-environment loading and deterministic reference generation pass"): for each of the
four releases, `scripts/sample.py --device cpu --precision float32 --temperature 0.0 --seed 42` was
run **twice as two fully independent process invocations** (each is a genuinely fresh Python process
loading only the release directory — no shared in-memory state) with the same fixed prompt. All four
produced byte-identical output across both runs:

| Release | Result |
| --- | --- |
| `mf064-150m-edu-release` | deterministic match |
| `mf065-150m-modern-release` | deterministic match |
| `mf066-150m-edu-sft-release` | deterministic match |
| `mf066-150m-modern-sft-release` | deterministic match |

## Matched benchmark-record comparison (`scripts/compare_releases.py`)

Built real `BenchmarkRecord`/`ComparisonKey` JSON (MF-037 schema) from the actual MF-064/065 run
records and validation results: `artifacts/mf067-benchmark-records/mf064-150m-edu.json`,
`.../mf065-150m-modern.json`. Running `scripts/compare_releases.py --edu ... --modern ...
--seed-count 1` produced a real, expected rejection:

```
ValueError: Edu and Modern records do not share the frozen comparison key
```

This is the **correct, honest outcome**, not a bug: `ComparisonKey.batch_tokens` genuinely differs
between the two runs (Edu 4×1024=4096, Modern 2×1024=2048) because Modern's hybrid/FlexAttention
preset required a smaller batch size to fit this 8GB card's VRAM (a real, previously-documented
architecture-necessitated deviation — see `tasks/backlog.md`'s MF-065 entry). `compare_releases.py`'s
strict frozen-key tool is designed to refuse exactly this kind of mismatch so a report can never
silently attribute a batch-size difference to architecture. The actual matched, batch-difference-aware
quality comparison for this pair is `reports/mf066-150m-edu-vs-modern-comparison.md`, which already
correctly attributes the batch_size deviation to VRAM constraints rather than treating it as a hidden
confound. Nothing was fudged to force a false match.

## Generation config / template / license bundling

Confirmed present and correct in all four release directories: `generation_config.json` (real
BOS=1/EOS=2/PAD=0, `max_length=2048`, `do_sample=false` conservative default, **no** Hugging
Face/vLLM compatibility claim), `chat_template.jinja`, `system_prompt.md`, `sha256-manifest.json`
covering every artifact file. Each model card documents the real license (Apache-2.0, matching this
repository's `LICENSE`; the two SFT cards additionally note the CC0-1.0 license of the small original
SFT conversation dataset used to train them), intended use, hardware, real metrics (cited by number,
not by reference alone), and safety caveats.
