# MiniFrontier V1 implementation backlog

This backlog freezes the build described in [`plan.md`](../plan.md): one raw-PyTorch decoder-only model, Edu and Modern presets, a 16K byte-level BPE tokenizer, from-scratch training, inference, evaluation, FIM, and small SFT. It deliberately excludes DeltaNet, MLA, MoE, RL, agents/tools, distributed training, custom kernels, and long-context production serving.

## Conventions

- **Priority:** P0 blocks the next milestone; P1 is required for V1; P2 is optional after V1.
- **State:** `Planned` initially. Move only dependency-free work to `Ready`.
- **Definition of done:** acceptance criteria pass, relevant tests are committed, documentation is updated, and no unrelated scope is added.
- **Evidence:** record exact commands, seed, config, environment, and metrics for training or benchmark tasks.
- **Hardware scheduling:** by user decision, CUDA/RTX execution evidence is collected after M10 implementation. CPU-verifiable code may proceed through M9, but tasks requiring trained 150M artifacts or GPU measurements remain open until those runs actually complete.

## Execution map

| Phase | IDs | Outcome |
| --- | --- | --- |
| Foundation | MF-001–006 | Reproducible package, tooling, presets, and governance |
| M0 Math | MF-007–012 | Tested primitives and tiny-batch learning |
| M1 Edu model | MF-013–019 | Complete 50M Edu Transformer |
| M2 Tokenizer/data | MF-020–027 | Real packed data and a 50M smoke run |
| M3 Inference/artifacts | MF-028–033 | KV cache, generation, checkpointing, and CLIs |
| Evaluation gate | MF-034–038 | Quality/efficiency baselines before experiments |
| M4 Modern model | MF-039–044 | GQA, QK-Norm, hybrid attention, global NoPE flag |
| M5 Performance | MF-045–050 | Reliable single-GPU training and profiling |
| M6 Coding/FIM | MF-051–054 | Provenanced code mixture and coding evaluations |
| M7 Muon lab | MF-055–057 | Fair AdamW-versus-Muon experiment |
| M8 SFT | MF-058–062 | Assistant-only SFT and usable chat path |
| M9 V1 release | MF-063–068 | Matched 150M Edu/Modern teaching release |
| M10 Optional scale | MF-069–070 | Evidence-based 350M/500M go/no-go |
| M11 Ecosystem adapters | MF-071–074 | Tested Hugging Face, vLLM/WSL2, and llama.cpp/GGUF distribution paths |

---

## Foundation

### MF-001 — Bootstrap the packaged Python project

- **Priority / state:** P0 / Done
- **Depends on:** none
- **Deliverable:** initialize the `uv` Python 3.12 project and lock the selected PyTorch backend using the safe `init.cmd`.
- **Acceptance:** `pyproject.toml`, `.python-version`, `uv.lock`, and importable `src/minifrontier/__init__.py` exist; `uv sync` succeeds; rerunning the bootstrap does not overwrite files.

### MF-002 — Configure developer tooling and CI

- **Priority / state:** P0 / Done
- **Depends on:** MF-001
- **Deliverable:** Ruff, pytest defaults, type-checking policy, and a minimal GitHub Actions workflow.
- **Acceptance:** lint and fast tests run locally and on supported CPU CI; generated data, environments, and training artifacts are ignored.
- **Status note:** `.github/workflows/ci.yml` was found missing from the working tree and git history entirely (2026-08-21) despite this task, MF-068, and prior evidence files describing it as already present and green — recreated (Windows runner, `uv sync --extra cpu --group dev`, Ruff lint/format, `pytest -m "not slow"`); `scripts/build_source_archive.py` (which hard-requires this file) now succeeds again. Also added `data/` to `.gitignore` (mirroring `artifacts/`), since real corpora/tokenizer/shard artifacts produced under MF-063 would otherwise be trackable by git.

### MF-003 — Create the final package and command skeleton

- **Priority / state:** P0 / Done
- **Depends on:** MF-001
- **Deliverable:** planned `src/`, `train/`, `scripts/`, `eval/`, `labs/`, `tests/`, `configs/`, `templates/`, and `artifacts/` layout.
- **Acceptance:** every planned module is import-safe or clearly marked as a stub; no placeholder silently reports success.

### MF-004 — Implement and validate model configuration

- **Priority / state:** P0 / Done
- **Depends on:** MF-003
- **Deliverable:** small `ModelConfig` plus TOML loading for 50M/150M Edu/Modern and optional 350M/500M Modern presets; attention implementations are `manual|sdpa|flex`.
- **Acceptance:** invalid head divisibility, context/window, position mode, attention implementation, or preset combinations fail clearly; parameter targets are documented and tested within tolerance.

### MF-005 — Define reproducibility and run metadata

- **Priority / state:** P1 / Done
- **Depends on:** MF-002, MF-004
- **Deliverable:** deterministic seeding helper and JSON run-record schema.
- **Acceptance:** records include config, seed, code revision when available, dependency versions, parameter count, token counts, timing, memory, losses, and hardware.

### MF-006 — Freeze repository and data licensing policy

- **Priority / state:** P1 / Done
- **Depends on:** MF-001
- **Deliverable:** selected repository license, third-party notices, dataset provenance rules, and a no-secrets/no-private-data policy.
- **Acceptance:** a public release has an explicit license; each data source must record source, revision, license, and content hash before training.

---

## M0 — Math works

### MF-007 — Implement RMSNorm test-first

- **Priority / state:** P0 / Done
- **Depends on:** MF-003, MF-004
- **Deliverable:** bias-free RMSNorm with explicit educational math.
- **Acceptance:** shape/dtype/device are preserved; output and gradients match an independent FP32 reference across representative shapes.

### MF-008 — Implement SwiGLU test-first

- **Priority / state:** P0 / Done
- **Depends on:** MF-003, MF-004
- **Deliverable:** three-projection, bias-free SwiGLU.
- **Acceptance:** output and gradients match the explicit `down(silu(gate(x)) * up(x))` reference; configured dimensions are correct.

### MF-009 — Implement causal and local mask helpers

- **Priority / state:** P0 / Done
- **Depends on:** MF-003, MF-004
- **Deliverable:** masks supporting arbitrary query offsets, full causal attention, and sliding windows.
- **Acceptance:** future keys and keys outside the window are blocked; offset/prefill/decode cases are covered without device mismatches; a mask needed by multiple layers is created once per model forward, not once per layer.

### MF-010 — Implement manual scaled dot-product attention

- **Priority / state:** P0 / Done
- **Depends on:** MF-009
- **Deliverable:** readable `QK^T / sqrt(d)`, mask, softmax, and `PV` path.
- **Acceptance:** known small tensors match hand-computed values; masked probabilities are zero; backward gradients are finite.

### MF-011 — Implement shifted language-model loss

- **Priority / state:** P0 / Done
- **Depends on:** MF-003
- **Deliverable:** next-token cross-entropy with optional ignore/loss mask support prepared for padding and SFT.
- **Acceptance:** shift direction is unit-tested; ignored tokens do not affect numerator or denominator; all-ignored input returns a differentiable zero without a GPU-synchronizing `.item()` hot-path guard.

### MF-012 — Prove the primitives learn a tiny batch

- **Priority / state:** P0 / Done
- **Depends on:** MF-007, MF-008, MF-010, MF-011
- **Deliverable:** deterministic CPU-friendly micro-model overfit test/lab.
- **Acceptance:** loss falls below a recorded threshold within a bounded step count and all gradients remain finite.

---

## M1 — Complete Edu Transformer

### MF-013 — Implement RoPE with external primitive parity

- **Priority / state:** P0 / Done
- **Depends on:** MF-004
- **Deliverable:** cached inverse frequencies and the documented rotate-half convention.
- **Acceptance:** shape/dtype/norm/position-zero/offset tests pass; values match a trusted Transformers RoPE primitive in a development-only parity test.

### MF-014 — Implement full causal MHA

- **Priority / state:** P0 / Done
- **Depends on:** MF-007, MF-009, MF-010, MF-013
- **Deliverable:** bias-free Q/K/V/O projections, RoPE, manual attention, and full causal MHA.
- **Acceptance:** tensor shapes and head transforms are tested; future-token perturbations cannot affect earlier outputs.

### MF-015 — Add the PyTorch SDPA execution path

- **Priority / state:** P0 / Done
- **Depends on:** MF-014
- **Deliverable:** optimized full-causal SDPA path selected by config/runtime flag while retaining manual teaching mode.
- **Acceptance:** full-sequence Edu SDPA passes `attn_mask=None, is_causal=True` so fused kernels remain eligible; cache decode uses correct offset semantics; manual and SDPA outputs/gradients agree in explicit FP32 fixtures and dtype-specific CUDA tests when available.

### MF-016 — Implement the pre-norm Transformer block

- **Priority / state:** P0 / Done
- **Depends on:** MF-007, MF-008, MF-015
- **Deliverable:** one-screen residual attention and FFN block.
- **Acceptance:** residual ordering is tested; shape is preserved; dropout defaults to zero and behaves only when configured.

### MF-017 — Implement the MiniFrontier language model

- **Priority / state:** P0 / Done
- **Depends on:** MF-004, MF-016
- **Deliverable:** token embeddings, N blocks, final RMSNorm, tied LM head, depth-aware initialization, and logits/loss API.
- **Acceptance:** weight tying is pointer-identical; base initialization scales with width and residual output projections scale by `1/sqrt(2*n_layers)`; forward/loss shapes and parameter counts match presets; sequence/position bounds fail without token-value `.item()` checks in the CUDA hot path.

### MF-018 — Complete Edu correctness and gradient tests

- **Priority / state:** P0 / Done
- **Depends on:** MF-017
- **Deliverable:** model-level tests for causality, finite loss, finite backward, device moves, saveable state, and deterministic eval.
- **Acceptance:** the fast CPU suite passes from a clean environment and covers the 50M preset through a reduced test config; manual/SDPA parity is explicitly pinned to FP32 so later BF16 autocast does not create a false failure.

### MF-019 — Prove Edu learning and validate 50M construction

- **Priority / state:** P0 / Done
- **Depends on:** MF-012, MF-018
- **Deliverable:** reproducible 100-example overfit using a CPU-sized faithful Edu config, plus exact construction of the frozen 50M Edu model.
- **Acceptance:** the 100-example CPU proof reaches `<1e-3` nats/token with a persisted run record and correct continuation sample; the 50M preset instantiates at exactly 53,361,152 trainable parameters. A full 50M training run is GPU work after the data/training pipeline exists.

---

## M2 — Tokenizer and real data

### MF-020 — Freeze tokenizer contract and token IDs

- **Priority / state:** P0 / Done
- **Depends on:** MF-004, MF-006
- **Deliverable:** 16,384-token byte-level BPE contract with BOS/EOS/PAD, chat, FIM, and tool-placeholder tokens reserved at training time.
- **Acceptance:** token strings, IDs, normalization, pre-tokenization, and document boundary behavior are documented and immutable for V1.

### MF-021 — Build the tokenizer training CLI

- **Priority / state:** P0 / Done
- **Depends on:** MF-020
- **Deliverable:** deterministic streaming trainer that emits `tokenizer.json` and metadata.
- **Acceptance:** fixed input/seed yields stable special-token IDs; arbitrary Unicode round-trips without unknown tokens.

### MF-022 — Test tokenizer correctness and quality

- **Priority / state:** P0 / Done
- **Depends on:** MF-021
- **Deliverable:** round-trip, special-token, boundary, code, multilingual, and compression smoke tests.
- **Acceptance:** all reserved tokens remain atomic; a versioned tokenizer fixture passes the suite.

### MF-023 — Define document and provenance schemas

- **Priority / state:** P0 / Done
- **Depends on:** MF-006
- **Deliverable:** normalized document record and source manifest for text/code data.
- **Acceptance:** schema captures source, revision, license, language, path/record ID, split, and content hash; invalid/missing provenance is rejected.

### MF-024 — Implement streaming source adapters

- **Priority / state:** P0 / Done
- **Depends on:** MF-023
- **Deliverable:** FineWeb-Edu streaming adapter and interface for local permissively licensed code manifests.
- **Acceptance:** sources can be bounded for tests, shuffled deterministically, resumed, and exercised without downloading a full corpus.

### MF-025 — Filter, deduplicate, tokenize, split, and pack

- **Priority / state:** P0 / Done
- **Depends on:** MF-022, MF-024
- **Deliverable:** end-to-end fixed-length packed sequence pipeline with EOS boundaries and train/validation isolation.
- **Acceptance:** no token loss/duplication across packing; duplicates do not cross splits; produced batches have correct dtype, shape, and labels.

### MF-026 — Build adversarial data-pipeline tests

- **Priority / state:** P0 / Done
- **Depends on:** MF-025
- **Deliverable:** tiny fixtures covering empty/long/bad-Unicode/duplicate documents, resume points, short final packs, and worker determinism.
- **Acceptance:** tests are network-free, deterministic, and detect split leakage and packing off-by-one errors.

### MF-027 — Run the 50M Edu real-data smoke train

- **Priority / state:** P0 / Done
- **Depends on:** MF-019, MF-026
- **Deliverable:** bounded full-50M packed-real-data forward/backward/checkpoint integration check on the CPU Dev Box.
- **Acceptance:** proves tokenizer/packer/model integration, finite loss/backward, checkpoint reload, metrics, and run records without making a quality claim. The deferred 1–5M-token FineWeb-Edu GPU continuation is explicitly tracked by MF-063 for the home RTX gate.

---

## M3 — Inference and artifacts

### MF-028 — Design and implement the KVCache API

- **Priority / state:** P0 / Done
- **Depends on:** MF-014, MF-017
- **Deliverable:** preallocated per-layer cache with `[batch, kv_heads, capacity, head_dim]`, logical length, positions, device, and dtype invariants.
- **Acceptance:** append/reset/reuse behavior, batch/device/dtype mismatches, non-contiguous positions, and overflow errors are tested; decode does not concatenate/reallocate the full history; training forward remains cache-free.

### MF-029 — Implement prefill and token-by-token decode

- **Priority / state:** P0 / Done
- **Depends on:** MF-028
- **Deliverable:** cached prefill/decode path with correct RoPE offsets and causal semantics.
- **Acceptance:** cached logits match full-forward logits over multiple prompt/decode/chunk lengths and batch sizes; RoPE positions never restart after prefill; full prefill uses fused-causal SDPA eligibility and single-token decode attends the complete cache without an incorrect square causal mask.

### MF-030 — Implement robust sampling

- **Priority / state:** P0 / Done
- **Depends on:** MF-029
- **Deliverable:** greedy, temperature, top-k, top-p, EOS, max-token, and seeded generation.
- **Acceptance:** temperature zero is deterministic; invalid arguments fail clearly; EOS can stop individual batch rows without corrupting others; requests exceeding model capacity fail explicitly rather than slicing context and restarting positions at zero.

### MF-031 — Implement training checkpoint and exact resume

- **Priority / state:** P0 / Done
- **Depends on:** MF-005, MF-017
- **Deliverable:** safetensors model plus optimizer, scheduler, RNG, data cursor, trainer state, and config.
- **Acceptance:** save/load preserves tied weights and logits; local pickle-based optimizer state is clearly trust-scoped; an interrupted deterministic test restores model/optimizer/scheduler/RNG/data cursor and matches uninterrupted loss/weights after resume.

### MF-032 — Implement published-model export/import

- **Priority / state:** P1 / Done
- **Depends on:** MF-021, MF-031
- **Deliverable:** release folder with model, config, tokenizer, tokenizer config, chat template, and model card skeleton.
- **Acceptance:** a fresh process loads only the release folder and reproduces reference logits/generation.

### MF-033 — Add sample and base-chat CLIs

- **Priority / state:** P1 / Done
- **Depends on:** MF-030, MF-032
- **Deliverable:** script entry points for prompt completion and interactive base-model inspection.
- **Acceptance:** `--help` works, seed/device/config are explicit, stdin/non-interactive use is testable, and errors do not expose secrets.

---

## Evaluation gate — required before architecture experiments

### MF-034 — Implement language-model validation metrics

- **Priority / state:** P0 / Done
- **Depends on:** MF-025, MF-031
- **Deliverable:** validation cross-entropy, perplexity, and bits-per-byte with weighted aggregation.
- **Acceptance:** hand-computed fixtures match; padding/partial batches do not bias metrics; BPB uses actual UTF-8 byte counts; evaluation is deterministic and avoids per-batch host synchronizations beyond metric collection boundaries.

### MF-035 — Build the lm-eval adapter and small suite

- **Priority / state:** P1 / Done
- **Depends on:** MF-030, MF-032
- **Deliverable:** adapter/config for ARC-Easy, HellaSwag, PIQA, and optional GSM8K.
- **Acceptance:** a tiny local smoke invocation completes and records harness version/task settings; failures are separated from low scores.

### MF-036 — Version the coding/FIM evaluation fixtures

- **Priority / state:** P1 / Done
- **Depends on:** MF-006, MF-020
- **Deliverable:** licensed or original fixtures for completions, FIM, syntax repair, and tiny unit-tested functions.
- **Acceptance:** no benchmark contamination from training manifests; expected syntax/compile/test scoring is unit-tested.

### MF-037 — Implement experiment and benchmark recording

- **Priority / state:** P0 / Done
- **Depends on:** MF-005, MF-034
- **Deliverable:** common runner that records quality, tokens/s, wall time, peak VRAM, inference throughput, and KV-cache bytes.
- **Acceptance:** records are schema-validated and comparable only when data, token budget, batch tokens, context, seed policy, and evaluation match.

### MF-038 — Publish the 50M Edu baseline scorecard

- **Priority / state:** P0 / Done
- **Depends on:** MF-027, MF-034, MF-035, MF-037
- **Deliverable:** baseline quality and efficiency report used by subsequent A/Bs.
- **Acceptance:** all available metrics, hardware, limitations, and raw run-record paths are documented; no unsupported quality claim is made.

---

## M4 — Modern architecture

### MF-039 — Implement GQA without hidden K/V expansion

- **Priority / state:** P0 / Done
- **Depends on:** MF-015, MF-028
- **Deliverable:** configurable query/KV heads and native SDPA GQA path, with explicit expansion only in the teaching/reference path.
- **Acceptance:** the manual reference expands K/V explicitly, while optimized full attention calls SDPA with `enable_gqa=True` and never repeats K/V in Python; output/gradient parity, projection/cache shapes, fused-causal prefill eligibility, and parameter/cache savings are tested.

### MF-040 — Add optional QK-Norm

- **Priority / state:** P0 / Done
- **Depends on:** MF-007, MF-039
- **Deliverable:** per-head RMS normalization of Q/K before RoPE.
- **Acceptance:** Edu defaults off, Modern defaults on; ordering and gradients are tested; disabling it reproduces the prior path.

### MF-041 — Implement the 3-local/1-global schedule with FlexAttention

- **Priority / state:** P0 / Done
- **Depends on:** MF-009, MF-039
- **Deliverable:** layer-indexed Local/Local/Local/Global attention with configurable window and a FlexAttention `mask_mod` optimized path; explicit masks and a full-history local KV cache remain the teaching/correctness reference.
- **Acceptance:** schedule and exact-window boundaries are tested for arbitrary layer counts; FlexAttention uses native GQA without Python K/V repetition and matches the manual reference where supported; fallback behavior is explicit; block masks are compiled/cached by shape rather than rebuilt per layer; the M4 reference cache may retain full history but local attention can never read outside its window or from unwritten storage. Bounded local-cache storage is an M5 optimization owned by MF-050, so M4 makes no local-cache memory-saving claim.

### MF-042 — Add the global NoPE experiment flag

- **Priority / state:** P1 / Done
- **Depends on:** MF-013, MF-041
- **Deliverable:** `global_position_encoding = rope|none`, retaining RoPE on local layers.
- **Acceptance:** RoPE-everywhere remains default; positions are skipped only in global attention under `none`; cached/full parity holds; attention-temperature tuning or logit soft-capping is not silently coupled to the flag, and long-context attention/logit magnitude plus finiteness are reported before any NoPE conclusion.

### MF-043 — Complete Modern correctness/cache tests

- **Priority / state:** P0 / Done
- **Depends on:** MF-040, MF-041, MF-042
- **Deliverable:** modern model suite covering GQA, local/global masks, QK-Norm, NoPE, and the full-history local-cache correctness policy.
- **Acceptance:** manual/optimized and cached/full parity pass for local and global layers under predeclared FP32 tolerances across prefill, single-token decode, and offset chunks; full and cached logits also have exact `argmax` agreement at every compared position; cache truncation never changes permitted attention; tests prove no out-of-window or unwritten cache slot can be read. BF16 behavior belongs to MF-046 and bounded/ring-cache parity belongs to MF-050.

### MF-044 — Overfit and compare 50M Modern

- **Priority / state:** P0 / Done
- **Depends on:** MF-038, MF-043
- **Deliverable:** matched 50M Edu/Modern correctness run and initial scorecard.
- **Acceptance:** Modern overfits the tiny set; the CPU comparison uses the same tokenizer/data/tokens/batch/context and reports parameter/cache differences without a speed claim. Hybrid efficiency conclusions are deferred to MF-050 and require an 8K+ GPU context run, including allocated/logical cache bytes and decode throughput.

---

## M5 — Single-GPU training performance

### MF-045 — Implement the canonical AdamW training loop

- **Priority / state:** P0 / Done
- **Depends on:** MF-027, MF-031
- **Deliverable:** explicit forward/loss/zero-grad/backward/clip/step/scheduler loop with warmup and cosine decay, consuming a bounded batch-provider interface rather than materializing a corpus.
- **Acceptance:** betas, weight decay exclusions, global gradient clipping, token accounting, validation cadence, and LR are config-driven and tested; warmup/cosine progress is defined in optimizer updates or consumed non-padding tokens, checkpointed explicitly, and has first/peak/final/off-by-one tests; an all-masked/no-target batch is rejected before an optimizer step; token dtype/range is validated on CPU before device transfer so invalid embedding indices cannot poison a CUDA context; debug tensor-value assertions and metric `.item()` calls stay outside per-layer/per-token and per-microbatch CUDA hot paths. This task uses bounded fixtures only—production shard creation/resume belongs to MF-047.

### MF-046 — Add BF16 autocast and gradient accumulation

- **Priority / state:** P0 / Done
- **Depends on:** MF-045
- **Deliverable:** capability-gated BF16 for training and cached inference, exact batch-token accumulation, explicit CLI precision selection, and a mixed-precision KV-cache dtype contract.
- **Acceptance:** FP32 CPU correctness remains; unsupported BF16 falls back clearly; accumulation matches an equivalent unsplit batch within tolerance; inference may cast weights or use autocast but cache storage is allocated lazily or explicitly from the actual projected K/V dtype, never inferred from the embedding output; sample/chat/evaluation paths expose `auto|float32|bfloat16` consistently; CPU-autocast coverage and CUDA BF16 full/cached numerical plus exact-argmax parity use tolerances fixed before measurement.
- **Status note:** CUDA evidence collected on the home RTX 2070 Super (2026-08-21, `reports/mf063-50m-gate.md`): `test_cuda_bfloat16_full_and_cached_logits_match_within_declared_tolerance` passes with a predeclared 5e-2 BF16 atol, exact argmax agreement, and lazy cache dtype confirmed non-inferred (adopts BF16 from the autocast projections). `test_cuda_gradient_accumulation_matches_unsplit_batch` also passes on CUDA. Collecting this evidence required fixing a real `KVCache` device-index bug (see MF-050 status note) that blocked every `--device cuda` cached forward pass; this GPU (Turing, CC 7.5) runs BF16 emulated rather than on native tensor cores, recorded honestly in `reports/mf050-rtx2070s-profile-matrix.md`.

### MF-047 — Make streaming training exactly resumable

- **Priority / state:** P0 / Done
- **Depends on:** MF-024, MF-025, MF-031, MF-046
- **Deliverable:** bounded-memory preprocessing into immutable, hashed, memory-mappable token/count shards plus a Windows-spawn-safe path-backed dataset, deterministic shard/row shuffle, checkpointed data-order state, and crash-safe publication.
- **Acceptance:** FineWeb-Edu is loaded directly by `scripts/prepare_data.py` without a giant intermediate JSONL from dataset `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, pinned revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; exact and versioned near-duplicate rules run before the deterministic hash split, and exact/near overlap with every frozen evaluation fixture is excluded or explicitly flagged; filtering/sanitization/dedup/contamination counters, thresholds, pipeline versions, and shard manifests are persisted without retaining rejected sensitive text; preprocessing never calls `list(...)` over the full corpus or stores a live one-shot generator in a worker dataset; production shards use separate atomic `.npy` arrays and a per-worker memory-map cache rather than reopening a ZIP/NPZ container per sample; each epoch uses a deterministic seed-derived shard permutation plus bounded within-shard row permutation, and checkpoints reject mismatched seed/shuffle policies while restoring the exact epoch/shard/row cursor. Interrupted and uninterrupted fixture runs consume identical document/token order and control state. Weight equality is exact only in a deterministic fixture; CUDA fused-backend runs record determinism settings and use a predeclared numerical tolerance rather than claiming universal bitwise reproducibility.

### MF-048 — Add optional `torch.compile`

- **Priority / state:** P1 / Done
- **Depends on:** MF-043, MF-046
- **Deliverable:** compile switch used only after eager tests pass, with graph-break diagnostics kept separate for training/prefill and mutation-heavy token decode.
- **Acceptance:** eager remains default; compiled outputs agree; cache mutation, dynamic sequence shapes, and EOS loop control are reported as graph-break/fallback causes rather than hidden; training/prefill compilation cannot imply that token decode compiled successfully; failures fall back or fail with an actionable message according to config.

### MF-049 — Add activation checkpointing by block

- **Priority / state:** P1 / Done
- **Depends on:** MF-046
- **Deliverable:** optional whole-block activation checkpointing for larger presets.
- **Acceptance:** losses/gradients agree within tolerance and a benchmark records the speed/memory tradeoff.
- **Status note:** CUDA correctness confirmed on the home RTX 2070 Super: `test_cuda_activation_checkpointing_matches_loss_and_gradients` passes (predeclared 1e-5 FP32 atol). Real speed/VRAM benchmark on 150M-edu with real FineWeb-Edu data, `reports/mf049-rtx2070s-checkpointing-benchmark.md`: at batch_size=1 (already fits), checkpointing costs ~22% throughput for ~28% VRAM savings (2746.9→2134.8 tok/s, 4.43→3.18 GB). At batch_size=4, eager training exceeds this 8GB card's VRAM and thrashes into Windows CUDA sysmem-fallback (100% GPU util, did not complete 60 updates in 400s+); with checkpointing the identical batch_size=4 run completes cleanly in 105.7s at 3.53GB peak — checkpointing is what makes that batch size usable at all on this card, not merely cheaper.

### MF-050 — Profile and tune the 24–32 GB GPU path

- **Priority / state:** P0 / Done
- **Depends on:** MF-037, MF-046, MF-047, MF-048, MF-049
- **Deliverable:** measured eager/SDPA/compile/checkpointing matrix for 50M and 150M plus a hardened single-stream inference path and long-context hybrid benchmark.
- **Acceptance:** generation computes only requested logit positions (`last-token` for ordinary prefill/decode) instead of materializing unused `[B,S,V]` logits, and output token storage avoids per-token whole-prefix concatenation; temperature is finite, top-k/top-p edge cases have hand-checked tests, filtering performs no redundant softmax, and non-finite model outputs fail diagnostically in an explicit debug/validation mode rather than being silently sanitized in the hot path. The full-history local cache remains the correctness reference; an optional bounded window/ring cache must track absolute RoPE positions, expose only initialized slots in chronological order, and match reference logits/argmax across wrap, chunk, reset, rollback, and mixed local/global layers. Single-token parity must cross at least two complete local-window wraps. This is a home-GPU gate: recommendations use recorded OOM-safe CUDA measurements; results separate failures from low throughput and record eager/compiled graph breaks, selected attention kernel/backend, determinism mode, precision/cache dtype, peak allocated/reserved VRAM, time-to-first-token, prefill tokens/s, inter-token latency, decode tokens/s, and allocated/logical cache bytes. Ring append/rollback allocation traffic is profiled explicitly before any correctness safeguard becomes optional. The 8K+ hybrid test uses a separately labeled performance-only config with unchanged weights and explicit context/window overrides, and makes no trained long-context quality claim. FlexAttention remains the required custom-mask path; where the pinned PyTorch build exposes first-party variable-length sliding GQA, it may be parity-tested and benchmarked as an optional backend rather than assumed faster. Measurements are single-stream or fixed-shape batch results, not vLLM/SGLang-style concurrency claims.
- **Status note:** runtime hardening, bounded mixed local/global ring storage, wrap/chunk/reset/rollback parity, single-token local SDPA decode dispatch, and the profile/report CLI are implemented. A first real home-GPU measurement pass ran on the RTX 2070 Super (2026-08-21, `reports/mf050-rtx2070s-profile-matrix.md`): 50M-edu/150M-edu/150M-modern, BF16 vs FP32, eager vs `torch.compile`, and an 8K-context hybrid-vs-full-history comparison. Key findings: this Turing GPU has no native BF16 tensor cores (FP32 measured faster than BF16, e.g. 1122 vs 609 tok/s prefill on 150M-edu) and `torch.compile` explicitly declines native BF16 compilation with no steady-state benefit; the 8K-context 150M-modern run exceeded the 8GB physical VRAM (peak ~10.9GB allocated) and silently fell back to slow Windows system-memory paging (33-37s TTFT) rather than crashing, while the ring-cache byte saving itself measured correctly (~3.4x). Collecting any of this required first fixing a real `KVCache` device-index bug: a bare `torch.device("cuda")` (no index) compared unequal to a real tensor's `torch.device("cuda", 0)`, so every documented `--device cuda` cached forward pass failed immediately — invisible to CPU-only CI. Fixed in `src/minifrontier/cache.py`. **Update (2026-08-26)**: the three remaining gaps are closed, all with real RTX 2070 Super measurements (`reports/mf050-rtx2070s-profile-matrix.md`'s 2026-08-26 update). (1) Manual-attention-path benchmarking: manual is consistently slower than SDPA on full-attention Edu models (10-39% slower prefill depending on precision), but on the hybrid Modern preset manual attention is **6.4x faster on prefill** than eager `attention_impl=auto` (which resolves local layers to eager FlexAttention) — confirming eager FlexAttention on this GPU is slower than the naive reference it exists to optimize. (2) Ring-cache allocation/rollback-traffic profiling: a one-off diagnostic driving `LayerKVCache.append` directly across 1223 single-token appends (>2 full 512-token wraps) shows the ring path is 4.4x slower per append than the linear path in wall-clock time, but allocator-level traffic (`allocation.all.current`, `num_alloc_retries`) is identical across ring/linear/committed/uncommitted — PyTorch's caching allocator reuses freed blocks for the unconditional rollback clones rather than issuing new `cudaMalloc` calls, so the real cost is extra kernel launches/copies (~25% higher peak active memory), not allocator churn; committing vs. never committing the rollback barely matters (0.439ms vs 0.452ms/append) since the clones are made unconditionally either way. (3) Batch-size sweep: 20 real runs (50M-edu/150M-edu x FP32/FP16 x batch 1/2/4/8/16, 128/32 prompt/decode) all stayed well under the 8GB danger zone (peak reserved VRAM ≤1.1GB, TTFT <0.2s throughout) and confirmed FP32 remains the faster *inference* precision at every batch size tested, though the FP32-vs-FP16 gap narrows at higher batch sizes. A synthesized recommendations section was added covering training precision (FP16), inference precision (FP32), `torch.compile` (not worth it here), the hybrid-prefill manual-vs-FlexAttention finding, the 8K-context VRAM ceiling, and available batch headroom. FlexAttention-vs-first-party-variable-length-sliding-GQA parity/benchmarking remains genuinely optional per `docs/IMPLEMENTATION_DECISIONS.md`'s own framing and was not pursued; nothing else in this task's acceptance criteria was found unmet, so MF-050 moves to Done.

---

## M6 — Coding and FIM

### MF-051 — Implement licensed code-corpus ingestion

- **Priority / state:** P1 / Done
- **Depends on:** MF-023, MF-024, MF-036
- **Deliverable:** manifest-driven ingestion and source-specific sanitization limited to approved permissive/public-domain sources.
- **Acceptance:** every record retains repo/revision/license/language/path/hash and admission-pipeline version; missing or disallowed licensing is rejected; reviewed filters cover secrets/credentials, the documented PII policy, malformed/binary/generated/vendor/minified content as applicable, exact/near evaluation contamination, and aggregate reason counts without retaining rejected sensitive text.

### MF-052 — Implement deterministic FIM transforms and mixing

- **Priority / state:** P1 / Done
- **Depends on:** MF-020, MF-025, MF-051
- **Deliverable:** prefix/suffix/middle serialization and configurable normal/FIM mixture.
- **Acceptance:** transform reconstructs the original text, handles boundaries/short samples, respects seed, and defaults to the frozen 10–20% experiment range.

### MF-053 — Run matched normal-versus-FIM code training

- **Priority / state:** P1 / Done
- **Depends on:** MF-044, MF-045, MF-047, MF-052
- **Deliverable:** controlled 50M Experiment 7 runs.
- **Acceptance:** only FIM mixing changes; token/data budgets and seeds are recorded; checkpoints and raw metrics are retained. A bounded CPU paired run proves integration but makes no effect-size claim; any coding-quality conclusion requires the later home-GPU budget recorded by MF-063.

### MF-054 — Score and report coding/FIM effects

- **Priority / state:** P1 / Done
- **Depends on:** MF-036, MF-053
- **Deliverable:** syntax-valid, compile, unit-test, and FIM exact/functional metrics.
- **Acceptance:** the scorer/report pipeline is fully exercised on versioned fixtures; reports include exact/near-contamination status, general-LM regression checks, variance/limitations, and links to raw records. CPU-only results are labeled engineering checks; effect claims wait for the MF-063 GPU gate.

---

## M7 — Optimizer laboratory

### MF-055 — Write the educational Muon reference

- **Priority / state:** P1 / Done
- **Depends on:** MF-007, MF-005
- **Deliverable:** small, non-production Newton–Schulz/Muon reference lab.
- **Acceptance:** math is annotated and tested on tiny 2-D matrices; production training cannot select this implementation accidentally.

### MF-056 — Integrate first-party Muon with AdamW partitioning

- **Priority / state:** P1 / Done
- **Depends on:** MF-045, MF-055
- **Deliverable:** Muon for eligible hidden 2-D matrices and AdamW for embeddings, norms, biases/scalars, and other ineligible parameters.
- **Acceptance:** each trainable parameter appears exactly once; partitioning is reported and tested; `match_rms_adamw` is supported when available.

### MF-057 — Run a fair AdamW-versus-Muon experiment

- **Priority / state:** P1 / Done
- **Depends on:** MF-037, MF-056
- **Deliverable:** matched-token A/B plus a small LR sweep per optimizer arm.
- **Acceptance:** comparison records optimizer-specific LR choices, throughput, VRAM, loss, and variance; conclusions do not rely on one unfair shared LR.
- **Status note:** the matched-token CPU engineering sweep uses separate LR grids and records partitioning/loss/throughput without an optimizer-quality claim. The post-M10 RTX rerun owns VRAM, variance, and scale conclusions.

---

## M8 — Supervised fine-tuning

### MF-058 — Implement and test the chat template

- **Priority / state:** P1 / Done
- **Depends on:** MF-020, MF-032
- **Deliverable:** simple Jinja template for system/user/assistant turns plus a concise, checked-in default system prompt for general and coding use.
- **Acceptance:** roles and EOS boundaries serialize deterministically; malformed role order fails; Jinja rendering, runtime chat encoding, and SFT serialization have one tested token contract across multi-turn and generation-prompt cases; the optional system prompt is capability-honest, short enough for the V1 context, and introduces no hidden-reasoning or tool protocol.

### MF-059 — Build SFT examples and assistant-only loss masks

- **Priority / state:** P1 / Done
- **Depends on:** MF-011, MF-025, MF-058
- **Deliverable:** conversation ingestion, packing, truncation, and assistant-token masks.
- **Acceptance:** system/user tokens never contribute to loss; assistant spans do; multi-turn and truncation edge cases are tested.

### MF-060 — Implement the small SFT loop

- **Priority / state:** P1 / Done
- **Depends on:** MF-045, MF-047, MF-059
- **Deliverable:** raw-PyTorch SFT stage reusing training/checkpoint infrastructure without Trainer/TRL.
- **Acceptance:** tiny examples overfit; batches use a deterministic epoch-dependent shuffle; exact resume restores and validates the shuffle seed/policy, epoch, cursor, and batch count; base checkpoint lineage and dataset subset are captured.

### MF-061 — Evaluate instruction following and regressions

- **Priority / state:** P1 / Done
- **Depends on:** MF-034, MF-054, MF-060
- **Deliverable:** small versioned prompt set plus base-versus-SFT evaluation.
- **Acceptance:** assistant formatting/instruction behavior, refusal/unknown handling, held-out functional code prompts, and language/code regressions are reported on versioned original or license-reviewed fixtures with contamination status; qualitative samples use fixed prompts/seeds, and no broad assistant-quality claim is inferred from the tiny suite.
- **Status note:** the versioned CC0 prompt set, transparent scorer, and base-versus-SFT runner are implemented; CPU fixtures validate the scorer. Model-quality results wait for the trained post-M10 artifacts.

### MF-062 — Complete the interactive chat path

- **Priority / state:** P1 / Done
- **Depends on:** MF-030, MF-033, MF-058, MF-060
- **Deliverable:** template-aware multi-turn CLI with bounded context.
- **Acceptance:** context truncation is explicit and preserves complete message/template boundaries; EOS works per turn; device/precision/seed and finite sampling controls are exposed; the release/default system prompt can be used, overridden inline/from a file, or disabled; the CLI documents that it is single-user educational inference rather than a continuous-batching server, and a CPU smoke test passes.

---

## M9 — Canonical 150M V1 release

### MF-063 — Freeze the canonical training protocol and budget

- **Priority / state:** P0 / Done
- **Depends on:** MF-054, MF-057, MF-061
- **Deliverable:** complete the deferred home-RTX 50M FineWeb-Edu and M4–M6 empirical gates, then freeze data mixture, batch tokens, context, optimizer/LR, seed policy, evaluation cadence, and compute budget for matched 150M runs.
- **Acceptance:** the pinned-revision 50M gate runs 1–5M real tokens and records decreasing loss, validation, CUDA dtype/cache parity, CPU-side token validation, eager/compile status, selected attention backend and determinism mode, throughput/peak VRAM, exact data/control-state resume plus tolerance-scoped numerical resume, cached samples, standard evaluation status, and the matched FIM follow-up; inference evidence includes last-token prefill, BF16 cached generation, bounded-local-cache parity when enabled, and time-to-first/inter-token latency. Infrastructure failures are separated from scores. A measured feasibility checkpoint then approves or adjusts the 150M plan. The target is at least 3B tokens unless the recorded go/no-go review changes it; if stopped while validation still improves, the release labels the checkpoint undertrained. If PDF/book material is selected, a dedicated clean-document adapter must fail closed on missing, ambiguous, or retrieval-only rights; exclude RAG overlap/frontmatter/citation/warning boilerplate; keep every book in one split using the source-PDF SHA-256 as parent identity; record source/converter/dependency/extraction hashes and page ranges; and pass tokenizer-fertility, corruption, deduplication, evaluation-contamination, source-cap, and mixture-ablation review. Native/local extraction is the default, and Gemini-derived text is excluded unless explicit terms and legal approval permit training and weight release; the external AGPL converter and raw books are never bundled into MiniFrontier.
- **Status note:** validated draft/frozen protocol tooling is implemented. The pinned-revision (`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`) home-RTX 50M FineWeb-Edu gate ran for real on the RTX 2070 Super (2026-08-21, full detail in `reports/mf063-50m-gate.md`): a real tokenizer trained on 3,000 real FineWeb-Edu documents, real packed shards (2,213,888 train tokens, within 1-5M), one full epoch (1,081 updates, BF16, CUDA) with loss decreasing 9.7→6.16 and validation cross-entropy decreasing 10.15→6.11 (perplexity 25,580→450), a genuinely-interrupted-and-resumed run on real hardware reproducing identical final loss across two different interruption points, a real BF16 cached sample from the trained checkpoint (honestly labeled undertrained — incoherent/repetitive, as expected at 2.2M tokens), a real end-to-end release export/audit of the trained checkpoint, and a real lm-eval smoke (ARC-Easy/HellaSwag/PIQA, n=20) scoring within noise of chance, as expected. Collecting this required first fixing a real `KVCache` CUDA device-index bug (see MF-050) that blocked every cached CUDA forward pass. The protocol remained `draft`: the matched FIM follow-up (an MF-053-style rerun) was not performed, and freezing without it, or with a sub-3B-token target, still requires the explicit `--approve-undertrained` decision the tooling fails closed on.
- **Status note (2026-08-26): FIM follow-up run and protocol frozen.** Real matched normal-vs-FIM training comparison on the RTX 2070 Super, full detail in `reports/mf063-50m-gate.md`'s "MF-063 FIM follow-up" section: a real permissively-licensed code manifest (this repository's own Apache-2.0 `src/minifrontier`/`scripts`/`train` source, 61 files, real GitHub remote + commit revision as provenance — no external licensing review needed), sanitized/admitted by `scripts/prepare_code.py` (57/61 admitted; 1 `generated`, 3 `personal_data` real rejections) and checked for zero overlap with `eval/fixtures/code_fim_v1.jsonl`. `scripts/apply_fim.py` produced matched baseline (0% transformed) and FIM (7/57, ≈12%, frozen-rate-consistent) manifests from the same 57 documents; both packed to identically 153 sequences / 156,672 tokens via the one frozen tokenizer. `scripts/compare_fim.py --config configs/50m-edu.toml --updates 300 --batch-size 2 --device cuda` ran both arms from identical starting weights, consumed identically 613,800 tokens each (enforced by the script), real Tensor Core FP16+GradScaler (this hardware's `"auto"` resolution, not emulated BF16): baseline final loss 1.612 vs FIM 1.668 — the FIM arm's higher loss/gradient-norm is the expected, honestly-reported cost of spending part of an identical token budget on the FIM rearrangement task in a run this short, not a production-scale effect-size claim (`compare_fim.py` itself labels this `"bounded_engineering_comparison"`). With this evidence, `scripts/freeze_protocol.py --status frozen --target-tokens 3000000000 --evidence <4 real files>` produced a valid frozen protocol (`artifacts/mf063-protocol/frozen.json`) — `target_tokens` stays at the full 3B V1 target (declaring the recipe the eventual MF-064/065 runs should follow, not a claim 3B tokens were trained); `undertrained_approved=false` is correct at that target per `TrainingProtocol`'s validation. The user has separately decided MF-064/065 will run at a smaller, explicitly-undertrained token budget "for now" rather than the full 3B — that will be recorded on those tasks' own evidence when they run, and does not reopen this frozen protocol. MF-063's full deliverable (freeze data mixture, batch tokens, context, optimizer/LR, seed policy, evaluation cadence, compute budget) is complete; state moves to Done.

### MF-064 — Train MiniFrontier-150M-Edu

- **Priority / state:** P0 / Done (reduced-budget pass; full 3B-token run deferred)
- **Depends on:** MF-063
- **Deliverable:** canonical Edu checkpoint, logs, and resumable training state.
- **Acceptance:** approved token budget completes without unresolved instability; best/final checkpoints and run metadata are retained.
- **Status note:** by explicit user decision (2026-08-26), this pass ran a real but deliberately reduced ~150M-token budget rather than the frozen protocol's 3B-token target — the full 3B-token run is deferred to later (this checkpoint is explicitly **undertrained** relative to that target, per MF-063's own acceptance criterion for a stopped-early run). Real data: a newly prepared `data/shards/mf064-150m-train` pool, 45,789,184 non-padding real FineWeb-Edu train tokens (pinned revision, documents 5,000+, past what the tokenizer and the MF-063 gate already consumed; 41,976/42,000 admitted, 2 exact + 22 near duplicates rejected), 436,224 validation tokens. `configs/150m-edu.toml`, FP16 (real Tensor Core, not emulated BF16 — see MF-075), batch_size=4, 36,620 updates, seed 42, on the RTX 2070 Super via `train/pretrain.py --device cuda`.
  - **Real result** (`artifacts/mf064-150m-edu/run.json`): 149,849,040 tokens consumed, train loss 9.7→**3.525**, 10,843.1 tok/s, wall time 13,819.8s (3.84h), peak allocated VRAM 6.65GB / reserved 7.24GB (no thrashing, matching the MF-049/075 real-data benchmark for this exact config).
  - **Validation** (`reports/mf064-150m-edu-validation.json`, same-seed untrained model as "before" baseline): cross-entropy 10.19→**3.79**, perplexity 26,545→**44.1**, bits/byte 3.43→**1.27** — a much larger real improvement than the MF-063 50M gate's 2.2M-token pass (which only reached CE 6.11 / PPL 450), consistent with ~68x more tokens and 3x more parameters.
  - **Inference evidence**: exported via `scripts/export.py --keep-source` (source checkpoint retained for possible later continuation toward 3B) to `artifacts/mf064-150m-edu-release`; a real CPU greedy sample (`scripts/sample.py --temperature 0.0`) from the trained checkpoint: *"The history of science shows that the world's population is growing at an alarming rate. The world's population is growing at an alarming rate..."* — coherent, grammatical English, but repetitive under greedy decoding, honestly consistent with a real-but-undertrained 150M checkpoint rather than a fluency claim.
  - Best/final checkpoints (`artifacts/mf064-150m-edu/checkpoint-*`, `final/`) are retained on disk for a possible later resume toward the frozen 3B target; this task's own scope (a real, evaluated, exported reduced-budget checkpoint) is complete.

### MF-065 — Train MiniFrontier-150M-Modern

- **Priority / state:** P0 / In progress
- **Depends on:** MF-063
- **Deliverable:** canonical Modern checkpoint under the matched protocol.
- **Acceptance:** tokenizer, data order/budget, batch tokens, context, and evaluation match Edu except documented architecture-specific settings.
- **Status note:** same reduced-budget decision and data pool as [[MF-064]] (`data/shards/mf064-150m-train`, tokenizer, seed, ~150M target tokens). Architecture-specific deviation, real-hardware-measured and necessary rather than arbitrary: batch_size=2 (not Edu's 4) — a real calibration (`artifacts/calibration-150m-modern-fp16-b*`) found batch_size=4 for this hybrid/FlexAttention preset pushed peak reserved VRAM to 9.25GB, over this 8GB card's physical budget, causing Windows CUDA sysmem-fallback thrashing (819 tok/s); batch_size=2 measured 4,224 tok/s at a safe 5.63GB peak. Activation checkpointing was also tried (batch 4 and 6) and did not beat batch_size=2's throughput, unlike Edu — Modern's bottleneck here is unfused eager FlexAttention compute, not VRAM headroom. ~73,242 updates targeting ~150M consumed tokens. Also explicitly labeled **undertrained** relative to the frozen 3B-token target, matching [[MF-064]].

### MF-066 — Produce the canonical teaching comparison

- **Priority / state:** P0 / Planned
- **Depends on:** MF-064, MF-065
- **Deliverable:** Edu-versus-Modern quality, training/inference throughput, VRAM, parameter, and KV-cache report.
- **Acceptance:** claims identify seed count and uncertainty; 3+ seeds are used for any strong causal conclusion or the result is labeled exploratory. The final evidence follows `docs/EVALUATION_RELEASE_GATE.md`: validation CE/perplexity/BPB; the historical baseline; compact reasoning/knowledge, instruction/chat, functional code/FIM, and trained-context retrieval tiers; exact task/revision/prompt/generation settings; and explicit contamination, unsafe-code sandbox, failure, and not-run states. ARC-Easy/HellaSwag/PIQA alone cannot support a general-chat or coding claim.

### MF-067 — Export release artifacts and model cards

- **Priority / state:** P0 / Planned
- **Depends on:** MF-032, MF-066
- **Deliverable:** load-tested Edu and Modern release directories with tokenizer, model and generation configs, weights, templates, cards, SHA-256 manifest, and evaluation results.
- **Acceptance:** fresh-environment loading and deterministic reference generation pass; `generation_config.json` records BOS/EOS/PAD IDs, context limit, and conservative sampling defaults without claiming Hugging Face/vLLM compatibility; the deterministic chat template and default system prompt are bundled; hashes cover every release artifact; model cards document supported precision/device paths, fixed-shape educational inference limits, data, license, intended use, hardware, metrics, and safety caveats.
- **Status note:** safe export, generation metadata, complete SHA-256 manifests, tamper detection, custom model cards, and matched-pair load auditing are implemented and CPU-tested. The task remains planned until MF-066 supplies the real canonical artifacts and results.

### MF-068 — Cut the GitHub-ready V1 release

- **Priority / state:** P0 / Planned
- **Depends on:** MF-002, MF-006, MF-067
- **Deliverable:** tagged repository with complete README, changelog/release notes, CI badge, reproducibility commands, and archived task state.
- **Acceptance:** the existing Windows CPU GitHub Actions workflow and the separately recorded home-GPU release smoke are green; clean clone setup/test/sample instructions pass; `docs/RESEARCH_SOURCE_REVIEW.md` is current and every bundled paper/reference archive has an explicit redistribution decision and notice, while raw third-party source ZIPs and papers are preferably replaced by pinned upstream links plus local-review hashes; the documented `scripts/build_source_archive.py` path includes `.github/workflows/ci.yml`, rejects unintended large files, and produces an archive containing no `__pycache__`, `*.pyc`, secrets, local paths, corpora, checkpoints, caches, duplicate third-party trees, or bundled research/reference archives; no unresolved P0/P1 tasks remain.

---

## M10 — Optional scale checks (not required for V1)

**User-scheduled implementation order:** MF-069 is a software/preflight task that may be completed
before the post-M10 RTX session. MF-070 owns the later measurements and remains gated on MF-068. No
estimate, CPU dry run, meta-device check, or simulated result substitutes for CUDA measurements or
trained checkpoints.

### MF-069 — Implement and preflight the optional scale-check harness

- **Priority / state:** P2 / Done
- **Depends on:** MF-045, MF-047, MF-048
- **Deliverable:** validated 350M/500M preset accounting, OOM-safe training/inference preflight commands, report schema, stop criteria, and CPU/meta-device dry-run coverage that reuses the M5 profiler/trainer.
- **Acceptance:** exact parameter counts and analytic lower-bound memory/cache estimates are labeled estimates; config/model construction, batch validation, checkpoint/resume wiring, report serialization, and deliberately tiny CPU paths pass without allocating full quality-scale activations; CUDA-only fields remain `unmeasured`, failures are recorded rather than converted into invented throughput, and no scale go/no-go or model-quality claim is produced.
- **Status note:** exact meta-device accounting, mixed local/global KV estimates, stop criteria, atomic reports, and a tiny checkpoint/resume smoke are implemented. The persisted CPU report leaves every CUDA/decision field unmeasured.

### MF-070 — Run the 350M scale check and decide whether to attempt 500M

- **Priority / state:** P2 / In progress
- **Depends on:** MF-068, MF-069
- **Deliverable:** measured 350M Modern memory/throughput/stability profile, explicit 500M go/no-go record, and—only if approved—a bounded 500M stretch run.
- **Acceptance:** the real single-GPU decision includes VRAM, throughput, time/cost, checkpointing needs, learning behavior, expected learning value, and stop criteria; failures remain first-class results and V1 artifacts are unchanged.
- **Status note:** trainer/profiler CUDA fields, measurement assembly, and the fail-closed decision tool are implemented. A decision cannot be emitted from CPU, failed, incomplete, or unmeasured evidence. The real 350M run and any 500M decision remain open.

---

## M11 — Ecosystem adapters (post-V1; not part of the neural core)

These tasks turn the frozen MiniFrontier artifacts into externally loadable formats. Uploading the
existing release directory to a model hub is useful file hosting, but does not by itself satisfy any
compatibility task below.

### MF-071 — Export a Transformers-compatible Hugging Face repository

- **Priority / state:** P2 / In progress
- **Depends on:** MF-067
- **Deliverable:** separate Hub-ready Edu and Modern repositories with a MiniFrontier `PretrainedConfig`, base model, causal-LM wrapper, tokenizer integration, safe weights, generation config, model card, and explicit `architectures`/`auto_map` metadata; the raw-PyTorch core remains framework-independent.
- **Acceptance:** in a clean pinned environment, `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, and `AutoTokenizer` load the local export and a pinned Hub revision with the documented remote-code trust setting; native-versus-Transformers logits and greedy tokens match for Edu and Modern fixtures, including GQA, QK-Norm, hybrid local/global attention, and the global-NoPE variant; tied weights, special/FIM/chat tokens, chat template/default system prompt, context limits, dtype, hashes, license, and revision are preserved. A repository upload without these tests is labeled hosted, not Transformers-compatible.
- **Status note:** the standalone configuration/model, safe export, tokenizer/chat metadata, complete hashes, local Auto-class loading, cached decode, and native FP32 logit/argmax parity are implemented and tested for tiny Edu/Modern/global-NoPE fixtures. Canonical 150M exports and pinned-Hub clean-environment tests remain gated on MF-067.

### MF-072 — Validate MiniFrontier with vLLM on Windows-hosted NVIDIA CUDA

- **Priority / state:** P2 / In progress
- **Depends on:** MF-050, MF-071
- **Deliverable:** a pinned vLLM integration that first targets the Transformers modeling backend and uses an out-of-tree model plugin only if the backend cannot express the frozen architecture, plus a Windows 11 + WSL2 run guide.
- **Acceptance:** the adapter follows vLLM's custom-model contract (`auto_map.AutoModel`, forwarded model/attention kwargs, `ALL_ATTENTION_FUNCTIONS`, and `_supports_attention_backend`); `vllm serve <hub-id> --model-impl transformers --trust-remote-code` or the documented plugin equivalent loads both canonical presets under WSL2 CUDA; prompt logprobs and greedy continuations match the native reference within dtype-specific tolerances across prefill/decode, GQA, hybrid layers, QK-Norm, and global NoPE; an OpenAI-compatible API smoke, BF16/cache behavior, version/revision, VRAM, TTFT, and decode throughput are recorded. Pinned plain-chat smokes cover Vercel AI SDK (`generateText`, `streamText`, messages, and an application-managed prompt loop), OpenCode, Cline, Roo Code, Kilo Code, and Aider with the real context limit and text-only/tool-disabled capability metadata. Transport success is reported separately from code-edit quality; V1 tool/function calling and native Win32 vLLM support are not claimed.
- **Status note:** the Transformers-backend structural contract, interleaved sliding/full layer metadata, WSL2 launch guide, native parity-fixture generator, fail-closed evidence schema, and completions/chat API harness are implemented. Actual WSL2 CUDA load/parity/performance and coding-client transports remain unmeasured.

### MF-073 — Add high-precision GGUF and llama.cpp architecture support

- **Priority / state:** P2 / In progress
- **Depends on:** MF-071
- **Deliverable:** a pinned upstream patch or maintained adapter implementing MiniFrontier conversion metadata, tensor mapping, tokenizer/chat/FIM metadata, model loading, and the llama.cpp compute graph for Edu and Modern.
- **Acceptance:** BF16 or F16 safetensors convert to GGUF without pretending the model is `llama` unless exact graph equivalence has been proven; the loader and graph preserve RoPE, GQA, QK-Norm, per-layer local/global attention, global NoPE, tied embeddings, special-token IDs, and context/window metadata; `llama-cli` and `llama-server` pass native-reference logit/greedy parity fixtures and a Windows 11 NVIDIA-CUDA smoke before quantization is attempted; upstream/adaptor revision and artifact hashes are pinned.
- **Status note:** pinned-checkout verification and fail-closed F16/BF16 conversion/report orchestration are implemented against the reviewed current upstream layout (`conversion/<model>.py`, `src/models/<model>.cpp`). They refuse conversion unless an actual MiniFrontier converter, GGUF constants/tensor mapping, architecture registration/declaration, and C++ graph exist. The reviewed baseline revision is recorded in `adapters/llama_cpp/upstream.json`; that upstream implementation and native Windows parity evidence remain open, and no Llama compatibility is claimed.

### MF-074 — Quantize, evaluate, and publish the 4-bit GGUF artifacts

- **Priority / state:** P2 / In progress
- **Depends on:** MF-038, MF-066, MF-073
- **Deliverable:** reproducible high-precision-GGUF-to-4-bit conversion, initially `Q4_K_M`, with separate Edu and Modern GGUF model repositories and usage cards.
- **Acceptance:** quantization starts from BF16/F16 GGUF rather than requantizing; the decision to use an importance matrix/calibration corpus is explicit, and any calibration data has a pinned revision, tokenizer/template, sampling recipe, license, hash manifest, and evaluation-contamination check; an uncalibrated comparison is retained when needed to isolate its effect. Windows 11 CUDA `llama-cli` and `llama-server` load and generate; size, peak RAM/VRAM, prompt/decode throughput, perplexity or validation loss delta, fixed-prompt greedy changes, LM/code regressions, tokenizer/template behavior, quantizer command/version, source checkpoint revision, and SHA-256 hashes are recorded. A file being 4-bit and loadable is not sufficient without the quality-regression report.
- **Status note:** a Q4_K_M-only candidate runner, GGUF validation, calibration-provenance gate, exact command/revision/hash report, and forced `publish_ready=false` state are implemented. Actual high-precision sources, quantized artifacts, Windows CUDA CLI/server smokes, and quality-regression reports remain open.

---

## Addendum — hardware-aware precision (discovered during MF-050/063 home-RTX evidence)

### MF-075 — Add real FP16 with GradScaler for non-native-BF16 CUDA hardware

- **Priority / state:** P1 / Done
- **Depends on:** MF-046
- **Deliverable:** a `"float16"` precision option alongside `auto|float32|bfloat16`, with automatic loss scaling for training, a hardware-aware `"auto"` policy that prefers genuine Tensor Core acceleration, and exact-resume-safe scaler state.
- **Acceptance:** `resolve_precision`'s `"auto"` mode prefers native BF16 (`torch.cuda.is_bf16_supported(including_emulation=False)`) over FP16 over FP32, rather than accepting PyTorch's default emulation-inclusive BF16 check; explicit `float16`/`bfloat16`/`float32` requests are still honored as asked; `float16` on CPU falls back to FP32 with a clear reason; `GradScaler` wraps the training backward/step path only when precision resolves to `float16` (a no-op for BF16/FP32); a skipped (inf/nan) optimizer step still advances the schedule/token counters, documented rather than silently surprising; the scaler's state round-trips through checkpoint save/resume exactly; every CLI that exposes `auto|float32|bfloat16` (`scripts/chat.py`, `scripts/eval.py`, `scripts/eval_sft.py`, `scripts/profile_model.py`, `scripts/sample.py`, `train/pretrain.py`, `train/sft.py`) adds `float16` consistently; CUDA FP16 full/cached logit parity uses a tolerance declared before measurement, mirroring the existing BF16 CUDA tests.
- **Status note:** discovered 2026-08-25 while reviewing `future-plan.md`: this project's RTX 2070 Super (Turing, CC 7.5) has no native BF16 Tensor Cores, so every prior `"auto"`/`"bfloat16"` GPU run (including all of `reports/mf050-*`, `reports/mf049-*`, `reports/mf063-*`) measured emulated, not accelerated, BF16. Implemented: `Precision` extended with `"float16"`; `resolve_precision`'s hardware-aware `"auto"` ordering; `GradScaler` wired into `train_updates` (unconditional calls that are transparent no-ops when disabled, verified against both plain AdamW and the Muon `CombinedOptimizer`); scaler state persisted via a new `TrainingState.grad_scaler_state` field, which round-trips through the *existing* generic checkpoint save/resume path with no changes needed to `checkpoint.py` or `pretrain.py`/`sft.py`; all 7 CLIs updated. Six new tests (2 CPU-mocked, 4 real-CUDA) all pass on this hardware. Re-measured on the real RTX 2070 Super (`reports/mf049-rtx2070s-checkpointing-benchmark.md`, `reports/mf050-rtx2070s-profile-matrix.md`, both updated 2026-08-25): **training throughput is ~2.7x faster under real FP16 than emulated BF16** (150M-edu batch_size=1: 7424 vs 2747 tok/s, using less VRAM too), and batch_size=4 — which never completed under BF16 eager (VRAM-thrashing) — completes cleanly at 10,968 tok/s under FP16 eager. Inference-only single-stream numbers are more mixed (FP32 still fastest for prefill/decode at this small scale) and are reported as measured, not rationalized further. FP16 is now the recommended training precision for this hardware going forward. This finding is now recorded in `AGENTS.md`'s Engineering rules as project-wide guidance.

### MF-076 — Stop training artifacts from accumulating unbounded disk usage

- **Priority / state:** P2 / Done
- **Depends on:** MF-031, MF-032
- **Deliverable:** a `--no-checkpoint` benchmark mode for `train/pretrain.py`/`train/sft.py` that never writes checkpoint files, and a verified-then-delete step in `scripts/export.py` so a training checkpoint's optimizer/RNG state does not outlive the release it was exported into.
- **Acceptance:** `--no-checkpoint` writes only `run.json`, no `checkpoint-*`/`final` directories, and does not change default behavior when omitted; `scripts/export.py` calls the new `minifrontier.release.verify_release` (manifest integrity, no pickle state, a real load-test forward pass with finite logits) before deleting anything, and only deletes the source checkpoint after that verification passes; an interrupted or corrupted export must never cost the only remaining copy of the trained weights; `--keep-source` opts out of deletion for users who want to keep resuming training from that exact checkpoint.
- **Status note:** discovered 2026-08-26 after a session of MF-046/049/050/063/075 benchmarking left `/artifacts` at ~22GB, almost entirely disposable intermediate checkpoints whose numbers were already captured in `reports/*.md`. Implemented and tested: `train/pretrain.py`/`train/sft.py --no-checkpoint` (new `tests/test_pretrain_cli.py`, the project's first functional — not just `--help` — test of a `train/*.py` entry point), `minifrontier.release.verify_release` (new unit test in `tests/test_release.py`), and `scripts/export.py`'s default-on auto-delete with `--keep-source` (verified end to end with a real CUDA checkpoint export, not just unit tests). `/artifacts` manually pruned from ~22GB to 816MB in the same session (kept the real MF-063 gate's `final/` checkpoint and its exported release).

### MF-077 — Harden cache compatibility, compile reporting, and controlled labs

- **Priority / state:** P0 / Done
- **Depends on:** MF-043, MF-048
- **Deliverable:** reject model/cache configuration mismatches, report lazy compilation only after a real compiled execution, and keep architecture/QK-Norm teaching comparisons controlled.
- **Acceptance:** a same-shaped Modern cache with a different local window fails before mutation; a backend failure raised during the first compiled execution is recorded and falls back to eager unless strict failure is requested; successful compilation is not reported before the first execution; architecture comparison labs use matched four-layer tiny fixtures; the QK-Norm lab starts common parameters identically and compares gradients over common parameters only; focused tests and the full fast suite pass.
- **Status note:** implemented and verified 2026-08-26. `KVCache.validate` now also compares the full `ModelConfig` (not just batch size/device) before any cache mutation, catching a same-shaped cache built from a different `local_window` (new `tests/test_cache.py::test_cache_rejects_same_shaped_model_with_different_local_window`). `minifrontier.compilation.maybe_compile` no longer reports `compiled=True` at wrap time: it returns a `_LazyCompileFallback` wrapper whose `CompileReport` is only marked compiled after the wrapped module's first real execution succeeds, and a backend failure on that first call is recorded and falls back to eager (or re-raises under `fail_on_error=True`), covered by two new `tests/test_training.py` cases with an injected failing backend. `labs/03_qk_norm.py` was reseeding once before building both models sequentially, so the qk_norm on/off models' shared weights were never actually identical; it now reseeds before each construction, asserts common parameters are bit-identical up front, and reports gradient norm over common parameters only, excluding QK-Norm's own scale parameters. `labs/02_mha_vs_gqa.py` already used matched `n_layers=4` fixtures on both sides, so no change was needed there. Verified: `pytest tests/test_cache.py tests/test_training.py` (30 passed), `pytest -m "not slow"` full fast suite (192 passed), `ruff check` on all changed files (clean), and `labs/03_qk_norm.py` run directly.
