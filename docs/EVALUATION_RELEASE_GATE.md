# V1 evaluation release gate

The original ARC-Easy, HellaSwag, and PIQA suite remains a cheap engineering baseline. It is not
sufficient evidence for a model described as a general, coding, or chat assistant. The canonical V1
comparison must report several small, explicitly scoped tiers.

## Base-language tier

- validation cross-entropy, perplexity, and bits per UTF-8 byte on the frozen validation split;
- HellaSwag for continuity with the 50M baseline;
- GSM8K as a small arithmetic/reasoning signal;
- fixed, versioned MMLU-Pro computer-science and mathematics subsets;
- a small GPQA-Diamond subset only as an exploratory floor measurement.

ARC-Easy and PIQA remain in the historical baseline table but cannot carry the final quality claim.
Every harness task, revision, few-shot setting, prompt format, sample limit, and failure is persisted.

## Coding tier

- the versioned MiniFrontier completion, FIM, syntax, compilation, and unit-test fixtures;
- MBPP and HumanEval only after license/revision review and only in an isolated code-execution
  environment with network disabled, resource/time limits, and disposable storage;
- exact and near-contamination checks against every admitted training source.

A generated string is not counted as functional code merely because it parses.

## Instruction/chat tier

- the checked-in assistant-format, instruction, refusal/unknown, and regression prompt set;
- an IFEval-compatible instruction-following subset once the evaluator applies the exact
  MiniFrontier chat template rather than treating the SFT model as a raw completion model;
- fixed system prompt, generation parameters, seeds, and complete qualitative samples.

The tiny prompt suite validates plumbing and exposes regressions; it does not establish broad safety
or assistant quality.

## Context tier

- deterministic needle/retrieval fixtures at the model's actually trained 1K/2K context;
- position-bucket reporting so failures near the end of context are visible;
- no 8K quality claim from the MF-050 performance-only override, which changes runtime capacity but
  does not train the model at 8K.

## Claims

Edu and Modern must use the same tokenizer, data order and budget, batch tokens, context, seeds,
generation settings, and evaluator revisions. Strong causal claims require at least three seeds;
otherwise the result is labeled exploratory. Infrastructure failures, unsafe-code tests not run, and
near-zero scores are separate states. Model cards publish all three honestly.
