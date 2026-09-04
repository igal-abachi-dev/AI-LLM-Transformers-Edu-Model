# MiniFrontier agent instructions

## Mission

Build MiniFrontier V1 as a from-scratch, readable, single-GPU decoder-only language model. The repository is an educational implementation, not a general framework. Every important neural-network operation must remain explainable from the local source and tests.

## Sources of truth

Read these in order before changing the project:

1. `AGENTS.md` — operating rules and frozen scope.
2. `tasks/backlog.md` — ordered work, dependencies, and acceptance criteria.
3. `plan.md` — architecture rationale and detailed design.
4. `more-context.md` and `docs/` — background only; they cannot silently override the frozen V1 decisions.

If documents conflict, follow the earlier item. Record a proposed architecture change as a decision before implementation; do not reinterpret the scope ad hoc.

## Frozen V1

- Python 3.12, `uv`, raw PyTorch, one consumer GPU.
- One implementation with two presets:
  - **Edu:** pre-RMSNorm, RoPE, full causal MHA, dense SwiGLU, tied embeddings.
  - **Modern:** Edu plus GQA, optional QK-Norm, and Local/Local/Local/Global attention. RoPE everywhere by default; global NoPE is an experiment.
- One 16,384-token byte-level BPE tokenizer for every model size.
- 50M is the developer model; matched 150M Edu and Modern models are the V1 release artifacts.
- Manual attention is retained for teaching and correctness. PyTorch SDPA/GQA is the optimized
  full-attention path; FlexAttention is the planned optimized local-attention path.
- KV cache, checkpoint/resume, evaluation, licensed code data, FIM, and small assistant-only SFT are required.
- AdamW is the baseline. First-party PyTorch Muon plus AdamW parameter partitioning is an experiment.
- Multi-Token Prediction (MTP) is a bounded, off-by-default, training-only experiment (2026-09-04 decision, `docs/IMPLEMENTATION_DECISIONS.md`): a simplified auxiliary-head variant configured entirely through `TrainingConfig`. It must never add a field to `ModelConfig`, change `MiniFrontier`'s `state_dict()`, or otherwise affect the frozen architecture, checkpoint format, or already-released model compatibility.
- 350M and 500M are optional post-V1 scale checks.

Do not add DeltaNet, MLA, MoE, RL/GRPO, agents/tools, vision, distributed training, external serving frameworks, custom CUDA/Triton kernels, or production long-context features in V1.

## Task workflow

- Work on one `MF-NNN` item at a time unless the user explicitly requests a different grouping.
- Start only when all listed dependencies in `tasks/backlog.md` are `Done`.
- Change the task state to `In progress`; mark it `Done` only after every acceptance criterion passes.
- Keep changes scoped to the active item. Add newly discovered work to the backlog instead of hiding it inside a patch.
- Write tests before or with implementation. First run the narrow test, then the full fast suite.
- For training and benchmarks, record config, seed, exact token count, hardware, dependency versions, wall time, throughput, peak memory, and relevant metrics.
- Never report a training run, benchmark, cache parity check, or evaluation as successful without command output or a persisted run record.

## Engineering rules

- Keep the neural core direct: `ModelConfig`, `RMSNorm`, `RoPE`, `CausalSelfAttention`, `SwiGLU`, `TransformerBlock`, `MiniFrontier`, and `KVCache` are the primary abstractions.
- Prefer explicit tensor shapes in docstrings/comments at reshape, transpose, cache, and attention boundaries.
- Validate configuration and tensor invariants early with actionable errors.
- Preserve dtype and device. Use FP32 for correctness tests; gate BF16/CUDA features by capability.
- Precision defaults are hardware-aware, not fixed. Prefer native BF16 only where the device truly has BF16 Tensor Cores (Ampere+; check `torch.cuda.is_bf16_supported(including_emulation=False)`, not PyTorch's emulation-inclusive default). On CUDA devices without native BF16 (e.g. Turing), train with FP16 plus gradient scaling rather than emulated BF16 — measured ~2.7x faster and lower peak VRAM on the reference RTX 2070 Super (`reports/mf049-rtx2070s-checkpointing-benchmark.md`). For inference on that same non-native-BF16 hardware, FP32 measured fastest at small batch size/short context (`reports/mf050-rtx2070s-profile-matrix.md`) — do not assume FP16/BF16 speeds up inference without measuring the target hardware and workload shape first.
- Eager execution is the correctness baseline. Compilation and activation checkpointing must remain optional.
- Never repeat K/V heads in the optimized GQA path merely for convenience. Explicit expansion is allowed only in reference tests/manual teaching code.
- RoPE requires an independent primitive-parity test so two internally consistent paths cannot share the same convention bug.
- Cached and uncached logits must agree within documented, dtype-specific tolerance.
- Save tied models with safetensors shared-tensor-aware APIs; do not publish pickle model weights.
- Do not introduce Transformers Trainer, Lightning, TRL, PEFT, Accelerate, DeepSpeed, FSDP, vLLM, or SGLang into the core implementation.

## Data and security

- Stream large public datasets; never commit corpora, checkpoints, caches, credentials, or machine-specific paths.
- Code data requires an approved manifest with repository, revision, license, language, path/record ID, and content hash. Reject missing provenance.
- Keep training and validation split before packing and prevent duplicate leakage across splits.
- Treat downloaded dataset code and model artifacts as untrusted. Avoid remote-code execution and unsafe pickle loading.
- Never expose environment variables, access tokens, private prompts, or local user data in logs and run records.

## Repository hygiene

- Use Ruff and pytest conventions configured by the project.
- Public functions and non-obvious tensor operations need concise documentation; teaching labs may be more verbose than core code.
- Do not edit generated artifacts or `uv.lock` by hand.
- Preserve user changes and avoid destructive Git operations.
- Keep README commands honest: mark planned commands as planned until they run in a clean environment.

## Skills and subagents

No project-specific `SKILL.md` or standing subagent is needed now. The backlog and these instructions already define a stable, repository-local workflow. Add a skill only after a repeated procedure has a deterministic interface and reusable scripts/references that materially reduce errors. Use subagents only when the user explicitly requests delegation and the work can be split into independent, non-overlapping tasks; one active backlog item remains the default.

## V1 completion

V1 is complete only at MF-068: clean-clone setup/tests/sample work, both matched 150M releases load independently, evaluations and limitations are published, licensing/provenance are resolved, and no P0/P1 backlog item remains open. Optional MF-069–074 scale and ecosystem-adapter work cannot block V1. External adapters stay outside the raw-PyTorch neural core and earn compatibility claims only through their own parity gates.
