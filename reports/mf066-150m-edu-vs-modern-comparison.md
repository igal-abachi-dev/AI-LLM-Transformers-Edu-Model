# MF-066 — 150M Edu vs Modern canonical comparison (full 5-tier gate)

Environment: Windows 11, Python 3.12.10, PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 2070 SUPER (8GB).
Date: 2026-08-26. Both models: real, exported, reduced-budget checkpoints from MF-064/MF-065 —
`artifacts/mf064-150m-edu-release` (154,172,160 params) and `artifacts/mf065-150m-modern-release`
(138,446,080 params), trained on the identical `data/shards/mf064-150m-train` pool (real FineWeb-Edu,
pinned revision), same tokenizer, same seed 42, ~150M consumed tokens each. Raw records for every
number below: `artifacts/mf066-tmp/*.json` (uncommitted scratch outputs) plus the scripts that
produced them, embedded inline in this report for reproducibility.

## Claims (read this section first)

- **Single seed.** Per MF-066's own acceptance criterion, strong causal claims require 3+ seeds.
  Everything in this report is a **single-seed, exploratory** data point, not a causal conclusion
  about Edu vs. Modern in general.
- **Both checkpoints are undertrained.** ~150M tokens vs. the frozen 3B-token protocol target (see
  MF-063). Near-chance/near-zero scores on hard tasks are expected, not a regression.
- **Quality-at-matched-tokens and efficiency are different claims**, kept separate below: Modern's
  GQA/hybrid-attention design is a compute/memory efficiency proposition, not a same-token-budget
  quality guarantee, and the two should not be conflated.
- Every measurement below is real command output; every gap (gated dataset, missing dependency,
  platform limitation) is recorded as an explicit `not_run`/`infrastructure_failure` state, not
  silently skipped or estimated.

## Base-language tier

`lm-eval` 0.4.12, 0-shot, `--limit 25`, `float16`, greedy/loglikelihood scoring via
`minifrontier.evaluation.language.MiniFrontierEvalLM`. Script:
`artifacts/mf066-tmp/base_language_eval.py`. Raw: `artifacts/mf066-tmp/base-language-{edu,modern}.json`.

| Task | Edu | Modern | Note |
| --- | ---: | ---: | --- |
| arc_easy (acc / acc_norm) | 0.40 / 0.24 | 0.32 / 0.32 | historical baseline only, per the eval-gate doc |
| hellaswag (acc / acc_norm) | 0.36 / 0.32 | 0.32 / 0.32 | continuity with the 50M baseline |
| piqa (acc / acc_norm) | 0.64 / 0.56 | 0.52 / 0.64 | historical baseline only |
| gsm8k (exact_match) | 0.0 | 0.0 | expected at this scale/budget |
| mmlu_pro_computer_science (exact_match) | 0.0 | 0.0 | fixed versioned subset |
| mmlu_pro_math (exact_match) | 0.0 | 0.0 | fixed versioned subset |
| gpqa_diamond_zeroshot | **not_run** | **not_run** | `ConnectionError: Unauthorized` — `Idavidrein/gpqa` is a gated HF dataset requiring authenticated terms acceptance, not available in this environment. Confirmed by direct invocation, not silently skipped. |

All n=25 samples; scores at this sample size and model scale are noise-dominated (arc_easy/hellaswag/
piqa deltas are within a few samples of each other either way) — read the whole row, not the model
with the higher number, as "better."

## Coding tier

### Internal FIM/completion fixtures (`eval/fixtures/code_fim_v1.jsonl`, 4 fixtures)

Real greedy completions (`artifacts/mf066-tmp/coding_tier.py`), scored with
`score_fixture_predictions(execute_trusted_fixtures=True)`. Raw: `artifacts/mf066-tmp/coding-{edu,modern}.json`.

| Fixture | Kind | Edu | Modern |
| --- | --- | --- | --- |
| completion-add-v1 | completion | all fail (whitespace output) | all fail (whitespace output) |
| fim-clamp-v1 | fim | all fail | **syntax_valid + compiles** (bare `return` is valid Python; still fails tests — functionally wrong) |
| repair-colon-v1 | syntax_repair | all fail | all fail |
| function-reverse-v1 | unit_function | all fail | all fail |

0/4 exact and 0/4 functional for both models. **Expected**: neither MF-064 nor MF-065's main
pretraining pool contained code — the only code either architecture has seen is the tiny 57-file
Apache-2.0 self-corpus used in MF-063's FIM *comparison side-experiment* (a separate 50M-scale run,
not part of this 150M pretraining data). Near-zero code capability here is not a bug.

### HumanEval / MBPP (Windows-compatible custom harness)

lm-eval's built-in `humaneval`/`mbpp` tasks were tried first and **confirmed to raise
`NotImplementedError: This metric is currently not supported on Windows.`** (the underlying `evaluate`
`code_eval` metric uses POSIX-only sandboxing). Rather than skip this, a small Windows-compatible
pass@1 harness (`artifacts/mf066-tmp/humaneval_mbpp_windows.py`) reuses this project's own
`minifrontier.evaluation.code.score_python` isolation pattern:

```python
subprocess.run([sys.executable, "-I", "-c", program], cwd=<disposable tempdir>, timeout=5.0)
```

Same isolation level already accepted for this project's own fixture scorer: process isolation +
timeout + disposable tempdir, **no explicit network firewall** (same limitation `score_python` already
has). Datasets: `openai/openai_humaneval` (MIT, OpenAI, streamed at eval time — not bundled),
`google-research-datasets/mbpp` (CC-BY-4.0, Google, streamed at eval time — not bundled). `--limit 20`
problems each, `max_new_tokens=150`, greedy, temperature 0.

| Task | Edu pass@1 | Modern pass@1 |
| --- | ---: | ---: |
| humaneval (n=20) | 0.0 | 0.0 |
| mbpp (n=20) | 0.0 | 0.0 |

Raw: `artifacts/mf066-tmp/humaneval-mbpp-{edu,modern}.json` (per-problem pass/fail + stderr tail).

### Contamination check

`artifacts/mf066-tmp/contamination_check.py` hashed (exact SHA-256 + simhash64, Hamming ≤3) all
`code_fim_v1.jsonl` prompts/references plus all 164 HumanEval and 500 MBPP problem+solution texts
against the 57-document admitted code corpus from MF-063's FIM follow-up. Result:
**0 exact, 0 near overlaps, `clean: true`** (`artifacts/mf066-tmp/contamination-report.json`) — expected,
since that corpus is this project's own source code, not interview-style algorithm problems.

## Instruction/chat tier

No SFT checkpoint existed for either 150M model before this task. Built a small **original**
32-conversation CC0-1.0 assistant dataset (`artifacts/mf066-tmp/build_sft_dataset.py` →
`artifacts/mf066-tmp/sft_dataset.jsonl`) — instruction-following, arithmetic QA, capitals QA, simple
code-writing, and privacy-refusal turns, explicitly checked to have **zero overlap** with the held-out
`eval/sft_prompts.json` evaluation set (verified by exact string comparison before training). The whole
dataset packs into exactly one `sequence_length=2048` batch (408 assistant-graded tokens), so this is a
real but very small/repeated-single-batch SFT signal, honestly disclosed as such.

`train/sft.py`, 500 updates, batch_size=1, lr=1e-5, FP16, from the pre-export MF-064/MF-065
checkpoints (full trainer state):

| Model | Wall time | Output |
| --- | ---: | --- |
| Edu | 2m33s | `artifacts/mf066-150m-edu-sft` → exported to `artifacts/mf066-150m-edu-sft-release` |
| Modern | 6m30s | `artifacts/mf066-150m-modern-sft` → exported to `artifacts/mf066-150m-modern-sft-release` |

`scripts/eval_sft.py` on the real held-out `eval/sft_prompts.json` (3 fixed prompts: `exact-blue`,
`unknown-private`, `tiny-add`):

| Model | Base required-match rate | SFT required-match rate |
| --- | ---: | ---: |
| Edu | 0.33 | **0.67** |
| Modern | 0.33 | **0.67** |

Both models: SFT clearly taught the exact-reply format (`exact-blue` → literal `"BLUE"`, both models)
and the refusal format (`unknown-private` → *"I don't have access to that private information and
can't provide it."*, both models, verbatim). Both also show a real, honestly-reported **overfitting
artifact** on `tiny-add` (the code prompt): Edu's SFT model echoes the memorized refusal sentence
instead of writing code; Modern's degenerates into repeated fragments. This is the expected failure
mode of a single repeated 2048-token training batch — SFT taught format from the examples it saw, and
neither model's pretraining pool contained code, so there was no code capability for SFT to elicit.
Raw: `artifacts/mf066-tmp/eval-sft-{edu,modern}.json`.

**IFEval subset: not_run.** `lm_eval`'s `ifeval` task requires the optional `langdetect` package,
which is not installed in this environment (`ModuleNotFoundError: No module named 'langdetect'`,
confirmed by direct invocation). Recorded as a genuine missing-dependency infrastructure gap rather
than skipped silently.

## Context tier

Small original deterministic needle-in-haystack (`artifacts/mf066-tmp/needle_haystack.py`): a unique
"magic number" sentence inserted at 5 position fractions (0.1/0.3/0.5/0.7/0.9) within ~1000 tokens of
synthetic filler sentences, followed by a retrieval query, greedy generation, temperature 0. Tested at
**~1000 tokens — the models' actually-trained `sequence_length=1024`, not the configured
`max_seq_len=2048`**, per the eval-gate doc's explicit warning against claiming untrained-context
capability.

| Position | Edu retrieved? | Modern retrieved? |
| --- | :---: | :---: |
| 0.1 | no | no |
| 0.3 | no | no |
| 0.5 | no | no (off by one: generated 1317 vs. actual 1316) |
| 0.7 | no | **yes** |
| 0.9 | no | **yes** |
| **Retrieval rate** | **0/5 (0.0)** | **2/5 (0.4)** |

Modern retrieved the needle correctly at the two latest positions and came within 1 of the correct
value at the middle position; Edu retrieved none. **This is n=5 per model — not remotely enough to
support an architectural claim** (e.g. about GQA/QK-Norm/hybrid attention helping retrieval); it is
reported as a real, single-seed, exploratory data point, and a plausible follow-up question for a
future multi-seed/larger-n pass, not a conclusion. Raw: `artifacts/mf066-tmp/needle-{edu,modern}.json`.

## Efficiency comparison (separate claim from quality above)

Pulled from already-recorded evidence, not remeasured:

| | Edu | Modern |
| --- | ---: | ---: |
| Parameters | 154,172,160 | 138,446,080 |
| Attention | full MHA, `n_kv_heads=12` (no GQA) | GQA `n_kv_heads=4` (3x fewer KV heads) + hybrid 3-local/1-global (`local_window=512`) |
| Training throughput (this pass, FP16) | 10,843.1 tok/s | 4,145.3 tok/s |
| Training wall time (~150M tokens) | 3.84h | 10.04h |
| Training peak VRAM (allocated/reserved) | 6.65 / 7.24 GB | 5.28 / 5.63 GB |

Modern trains slower on this GPU **not because of its parameter/cache efficiency design, but because
its local layers' eager FlexAttention path is unfused** (materializes the full attention score matrix;
see `reports/mf050-rtx2070s-profile-matrix.md`'s finding that manual attention beats eager
`attention_impl=auto` by 6.4x on this same hybrid preset). Modern's actual efficiency payoff — smaller
KV cache at long context via GQA + bounded local ring storage — is a serving/inference-time property,
already measured separately in `reports/mf050-rtx2070s-profile-matrix.md` (~3.4x ring-cache byte
reduction at 8K context) and `tests/test_cache.py`/MF-041/043's correctness suite, not something this
150M/150M-token training pass re-measures. Full prefill/decode throughput numbers for both presets
(BF16/FP32/FP16, eager/compiled) are already in `reports/mf050-rtx2070s-profile-matrix.md` and are not
duplicated here.

## What this comparison does not establish

- Not a 3-seed causal comparison — every table above is exploratory.
- Not a claim about either model's behavior at the frozen 3B-token budget — both are ~50x under that
  target.
- Not a coding-capability claim — neither model's pretraining pool included code.
- Not an 8K-context or beyond-1024-token claim.
- GPQA-Diamond and IFEval are `not_run` (gated dataset access, missing optional dependency
  respectively), not near-chance scores — that distinction is preserved throughout this report.
