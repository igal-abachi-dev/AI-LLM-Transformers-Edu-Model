# Evaluation policy

MiniFrontier separates engineering validity from model quality.

- `scripts/eval.py` computes token-weighted cross-entropy/perplexity and byte-weighted BPB,
  locally smoke-tests the lm-eval adapter, and optionally runs ARC-Easy, HellaSwag, PIQA, and
  GSM8K through `lm-eval`.
- `scripts/smoke_50m.py` is the CPU-safe M2/M3/evaluation integration gate. Its one optimizer
  step proves packed real text, the frozen 50M model, backward, checkpoint reload, cached
  generation, and record generation compose correctly. It does not claim useful quality.
- `eval/fixtures/code_fim_v1.jsonl` contains original Apache-2.0 evaluation-only fixtures.
  Running fixture tests is explicit because subprocess isolation is a reliability boundary,
  not a security sandbox.
- Benchmark records are comparable only when their full `ComparisonKey` matches: data,
  tokenizer, token budget, batch tokens, context, seed policy, and evaluation identity.

The home RTX gate should run 1–5M FineWeb-Edu tokens and the standard task suite before any
architecture experiment uses the 50M baseline as evidence.

The standard task suite is a historical baseline, not the final assistant/coding release gate.
Canonical M9 claims must also follow
[`EVALUATION_RELEASE_GATE.md`](EVALUATION_RELEASE_GATE.md), which defines the compact reasoning,
functional-code/FIM, instruction/chat, and trained-context retrieval tiers; exact revision and
prompt records; contamination checks; isolated code-execution policy; and explicit failure/not-run
states. Tasks without a correct MiniFrontier chat-template adapter are reported as not run rather
than evaluated with a semantically different prompt format.
