# From a finished pretraining run to a Hugging Face upload

A real, ordered procedure for what happens between a pretraining run (e.g. `MF-070`) finishing
its last update and a finished artifact being published on the Hugging Face Hub. Written
2026-10-01 while `MF-070` was still ~82% through training, so this project has the sequence
settled *before* the run finishes rather than figuring it out under time pressure. Nothing here
is new policy -- every step cites the existing tool, task, or precedent it reuses.

## 1. Training completion

The run reaches its full configured `--updates` (`MF-070`: 732,422, the WSD schedule's own real
decay-to-completion point). No separate action beyond letting the run finish -- `checkpoint-*`
directories and `train-*.log` already carry everything later steps need.

## 2. Real held-out validation

Not yet available for `MF-070` specifically: `run_mf070_150m_release.ps1`'s own launch command
has no `--validation-interval`, so only plain training loss has been logged throughout the run
(a real, disclosed gap recorded in `MF-070`'s own backlog entry). Run `scripts/eval.py` against
the final checkpoint to get real cross-entropy, perplexity, and bits-per-UTF-8-byte on the frozen
validation split -- the actual overfitting check this project relies on (train loss trending down
in step with held-out CE/PPL/BPB, the same pattern `MF-065`'s own 1B-token run already passed).

## 3. The 5-tier eval gate

`docs/EVALUATION_RELEASE_GATE.md` is the canonical definition; `MF-066` already ran this once in
full on the prior (pre-3B-token) 150M generation, so this is a repeat of a proven procedure, not
new process:

- **Base-language tier**: the step-2 validation metrics, HellaSwag, GSM8K (`scripts/eval.py
  --include-gsm8k`), fixed MMLU-Pro CS/math subsets, a small GPQA-Diamond floor check.
- **Coding tier**: `scripts/eval_code.py` against the versioned completion/FIM/syntax fixtures --
  now including the TypeScript fixture added in `MF-165` specifically because the coding tier had
  no coverage in the one language where this project's own live-sample investigation found real
  cross-language bleed-through. Pass `--tokenizer` to also get the `MF-165` repetition/degeneracy
  signal (`distinct_n`/`repeated_ngram_fraction`) alongside syntax/compile/functional rates --
  off by default, worth turning on for this real run given what was found mid-training.
- **Context tier**: deterministic needle/retrieval fixtures at the model's actually-trained
  context length.
- **Instruction/chat tier**: not applicable to the base checkpoint -- this tier is scored once an
  SFT variant exists (step 5).

## 4. Export + fresh-environment verification

`scripts/export.py --checkpoint <final> --tokenizer data/tokenizer --output <release-dir>
--model-card` -- the same tool `MF-067` already proved end-to-end on the prior generation
(`reports/mf067-release-verification.md`). `--model-card` generates the model card directly from
the export call; no separate card-building script is needed for a model (unlike the dataset card,
which needed `scripts/build_dataset_card.py`). Verification repeats `MF-067`'s own real checks:
`minifrontier.release.verify_release` (manifest integrity, no pickle state, finite load-test
logits), and two independent fresh-process `scripts/sample.py` invocations at a fixed
prompt/seed/`temperature=0` confirming byte-identical output.

## 5. SFT -- a separate, additionally-published release, not a gate on the base one

Real precedent already exists: `MF-066` published **four** releases for the prior generation --
base Edu, base Modern, and a clearly-labeled SFT variant of each, never conflated with the base
(`artifacts/mf066-150m-edu-sft-release`, `artifacts/mf066-150m-modern-sft-release`). The same
shape applies here: the base `MF-070` release finishes steps 2-4 on its own first. Separately,
`MF-123`'s real reasoning-demonstration dataset (`data/sft/gsm8k-reasoning-v1.jsonl`, 7,473
provenance-tracked GSM8K train-split examples, already built and verified while the base run was
still training) feeds `train/sft.py` against the finished base checkpoint. The resulting SFT
checkpoint gets its own pass through the instruction/chat tier (`MF-061`'s existing
base-vs-SFT comparison pattern) and its own export + model card via step 4's same tooling. Both
the base and the SFT'd model end up as separate, clearly labeled artifacts -- matching how real
released model families ship `Base` and `Instruct` as distinct repositories rather than one
replacing the other. Only once an SFT release exists does `chat.py` become the right tool to test
with (a base-only checkpoint has never seen the `<|system|>`/`<|user|>`/`<|assistant|>` role
markers in any trained way -- `sample.py` remains correct for the base release).

## 6. The Transformers-compatibility fork, honestly disclosed

`MF-071` ("Export a Transformers-compatible Hugging Face repository") is **In progress**, not
Done -- its own acceptance text is explicit: *"A repository upload without these tests is
labeled hosted, not Transformers-compatible."* The standalone config/model, safe export,
tokenizer/chat metadata, local `Auto*`-class loading, and native-vs-Transformers logit/argmax
parity are implemented and tested only on tiny synthetic Edu/Modern/global-NoPE fixtures --
**never yet run on a real canonical 150M checkpoint**, which `MF-070`'s own release would be the
first real opportunity to do. Two honest options when this step is reached, not a silent default:

- **Upload step 4's plain export now**, labeled plainly as a hosted repository, not a
  Transformers-compatible one -- faster, real precedent (step 4 tooling is already proven).
- **Finish `MF-071`'s real parity pass on the actual `MF-070` checkpoint first** -- more work, but
  earns the stronger, accurate `AutoModel`/`AutoModelForCausalLM`-loadable claim.

## 7. The upload itself

One real, honest unknown worth flagging rather than assuming away: this project has a fully
proven Hugging Face **dataset** upload (the real 5-source `MF-070` training mixture, including
the real NTFS-junction lesson learned mid-upload -- see `MF-070`'s own backlog thread), but no
record of a trained **model** ever actually being pushed to the Hub before -- every prior release
(`MF-064`-`MF-068`) was exported locally with a model card written, never confirmed uploaded.
The model upload itself, whenever it happens, is a genuine first for this project, not a repeat
of an already-tested path -- budget real time for it to surface its own new lessons, the same way
the dataset upload did.

## Related

- `docs/EVALUATION_RELEASE_GATE.md` -- the canonical definition step 3 reuses.
- `reports/mf067-release-verification.md` -- the real precedent step 4's tooling already passed
  once.
- `tasks/backlog.md`'s `MF-070`/`MF-071`/`MF-123`/`MF-165` entries -- the real task state each
  step above depends on; this file is a sequencing map, not a replacement for those entries'
  own detail.
