# Implementation decisions

This file records decisions that refine the frozen architecture without changing V1 scope.

## 2026-08-17 — M1 hardening before real data

- Full-sequence Edu SDPA uses `attn_mask=None, is_causal=True` so PyTorch may select a fused backend. Explicit quadratic masks remain in the manual teaching path and in offset/chunk reference cases that require them.
- Any explicit mask shared by layers is built once in `MiniFrontier.forward` and passed through blocks. Modern local attention will use FlexAttention where supported rather than pretending masked SDPA is a sliding-window fused kernel.
- `attention_impl` reserves `flex` now; Modern MF-041 owns its implementation and fallback policy.
- Initialization uses a width-aware base standard deviation and scales attention/FFN residual output projections by `1/sqrt(2*n_layers)`.
- CUDA hot paths avoid tensor-value `.item()` validation. Shape, Python position, and configuration checks remain; expensive value assertions belong in tests or explicit debug tooling.
- The M1 overfit correctness gate is `<1e-3` nats/token, not merely a large percentage reduction.
- M3 generation rejects capacity overflow. It never slices a shifted window and silently restarts RoPE positions.
- Hybrid-attention performance conclusions require 8K+ context. At 2K with a 512 window, results are reported only as the cost side of the tradeoff.

## Hardware-tiered evidence

The Azure Dev Box is the correctness environment and uses the CPU PyTorch extra. Full architecture construction, small faithful overfits, bounded 50M forward/backward smoke checks, tokenization, packing, checkpointing, and evaluation adapters must work there. CUDA fused-kernel selection, meaningful token-budget training, and performance claims require the home RTX environment and are recorded as separate evidence rather than inferred from CPU behavior.

## 2026-08-18 — Pre-M4 data, cache, and measurement contracts

- The architecture remains frozen; this review adds implementation gates, not another model
  redesign.
- The FineWeb-Edu adapter and production preprocessing use dataset `HuggingFaceFW/fineweb-edu`, config
  `sample-10BT`, and immutable revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The config name is never recorded as the source
  revision.
- The generic M2 filter is a structural smoke-stage only. Source-specific sanitization and
  aggregate reason counts are mandatory before real training and are owned by MF-047/MF-051.
- MF-045 consumes bounded providers; MF-047 owns immutable token shards and a path-backed,
  Windows-spawn-safe loader. A live one-shot generator and whole-corpus `list(...)` are forbidden
  for production preprocessing.
- A KV cache's dtype is the actual projected K/V dtype. Under autocast it must not be inferred
  from the FP32 embedding output. MF-046 owns CPU-autocast and CUDA-BF16 regression coverage.
- Token IDs are range-checked on CPU before transfer to CUDA; the model hot path relies on the
  embedding contract and avoids a synchronizing device-side validation.
- Cache parity uses both fixed dtype-specific numerical tolerances and exact per-position argmax
  agreement. Tolerances are chosen before a run and are not loosened to make a result pass.
- `torch.empty` cache storage remains valid only while the logical-length invariant makes
  unwritten slots unreachable. Any local wrapping/ring design must initialize or mask every
  readable slot and prove that invariant; zero-filled storage is not a substitute for correct
  bounds.
- CPU runs can close correctness/integration gates, but throughput, VRAM, compile, BF16, hybrid
  efficiency, and FIM effect claims remain home-GPU evidence under MF-050/MF-063.

## 2026-08-18 — Inference reference review before M4

- The local Grok-1, vLLM, SGLang, Llama 4, and teaching references were reviewed and are recorded
  with exact archive hashes in [`INFERENCE_REFERENCE_REVIEW.md`](INFERENCE_REFERENCE_REVIEW.md).
- M3 remains the eager correctness baseline. Its fixed-shape, single-stream cache/decode design is
  appropriate for V1; serving-engine schedulers, paged/Radix caches, speculative decoding,
  quantization, offload, and concurrency are not added.
- M4 local attention first uses a full-history cache with an exact window mask. MF-050 may add a
  bounded ring/window cache only as a tested optimization against that reference. This separates
  semantic correctness from allocator complexity and prevents a premature memory-saving claim.
- MF-046 owns cached-inference precision and derives storage dtype from projected K/V. MF-050 owns
  last-token prefill logits, non-quadratic generated-token storage, sampling edge hardening, and
  time-to-first/inter-token latency measurements.
- Global NoPE remains a one-variable experiment. Llama-style attention-temperature tuning and
  Grok-style logit soft-capping are not silently bundled with it; stability metrics are reported.
- Final releases include generation metadata and artifact hashes but make no external-runtime
  compatibility claim without a separately tested adapter.

## 2026-08-18 — External distribution remains separate and test-gated

- The native MF-067 artifact is correct for `minifrontier.checkpoint.load_release`, and the same
  files may be hosted on Hugging Face, but hosting alone does not make them Transformers-, vLLM-,
  or GGUF-compatible.
- Post-V1 MF-071 adds a real MiniFrontier Transformers configuration/model/tokenizer export rather
  than falsely declaring the architecture to be Llama. It must preserve both Edu and all frozen
  Modern semantics and prove logit/greedy parity.
- MF-072 targets vLLM's Transformers modeling backend first, with an out-of-tree adapter only when
  necessary. Windows-hosted NVIDIA validation uses WSL2; native Win32 vLLM is not promised.
- MF-073 implements and validates high-precision GGUF/llama.cpp support before MF-074 performs
  four-bit quantization and quality evaluation. Native Windows CUDA is part of the llama.cpp gate.
- These adapters remain outside the neural core and cannot retroactively change the canonical
  checkpoints or block the educational V1 release.

## 2026-08-18 — Full research-source audit

- The complete local PDF/context inventory, hashes, relevance, and public-release disposition are
  recorded in [`RESEARCH_SOURCE_REVIEW.md`](RESEARCH_SOURCE_REVIEW.md). `more-context.md` and the
  transcript appended to `plan.md` are non-normative; the backlog remains authoritative.
- The local arXiv `2201.11903` file is Chain-of-Thought, not the Chinchilla scaling paper. The
  canonical 150M target is at least 3B tokens subject to MF-063 feasibility, and an early stop with
  improving validation is labeled undertrained rather than presented as a settled quality result.
- Hybrid efficiency uses an unchanged-weight, separately labeled 8K+ performance config. The
  ordinary 1K/2K presets teach semantics and do not support a long-context quality/efficiency claim.
- MF-047 adds versioned near-deduplication and evaluation-contamination checks for the release
  corpus. Exact data/control resume is separated from backend-dependent CUDA numerical replay.
- MF-045 makes scheduler progress and all-masked-batch behavior explicit. MF-050 records the actual
  selected attention backend/determinism mode and may benchmark PyTorch's first-party variable-
  length sliding GQA only as an optional parity-proven path.
- MF-074 records the provenance and contamination status of any GGUF importance-matrix/calibration
  data and retains a comparison sufficient to isolate calibration from quantization.

## 2026-08-18 — M4–M6 implementation freeze

- Modern keeps the frozen design: compact-K/V GQA, optional per-head QK RMSNorm before RoPE,
  Local/Local/Local/Global attention, and a global-only NoPE switch. `attention_impl=auto` maps
  local layers to FlexAttention and global layers to SDPA; CPU training fixtures explicitly use
  masked SDPA because FlexAttention backward is CUDA-only in the pinned build.
- Manual attention expands K/V only as a teaching reference. SDPA and FlexAttention receive
  compact K/V and first-party GQA flags. Flex block masks are cached by device and geometry.
- M4 local caches retain full history as the correctness baseline. MF-050 adds optional bounded
  per-local-layer ring storage while global layers retain full history; absolute positions remain
  separate from storage slots, chronological initialized reads survive wrap, and transaction
  rollback restores overwritten slots after a failed forward.
- Local training/prefill keeps FlexAttention. Cached single-token local decode dispatches to SDPA;
  with bounded storage every retained key is legal, avoiding both a per-token Flex block-mask cache
  entry and unnecessary window-mask construction.
- Production token shards are separate hashed `.npy` token/count arrays with one memory map cached
  per worker. Data order uses deterministic epoch shard permutations and bounded within-shard row
  permutations, with seed/policy/epoch/shard/row state validated during exact resume.
- The canonical trainer schedules by completed optimizer update, accumulates exact target-token
  loss sums, validates token ranges on CPU, and serializes model, optimizer, scheduler, RNG, and
  immutable-shard cursor state. Checkpoint model/training configuration mismatches fail closed.
- Inference cache allocation is lazy from projected K/V dtype. Generation requests only the last
  logit, preallocates output tokens, validates finite sampling parameters, and exposes optional
  non-finite-logit diagnostics.
- `scripts/profile_model.py` is intentionally not named `profile.py`; that filename shadows the
  Python standard library and breaks PyTorch determinism/compile imports when scripts run directly.
- Code FIM transforms preserve `parent_content_hash` and a versioned transform identifier. Dataset
  splits use this stable parent identity, preventing the baseline and FIM arms from silently moving
  the same file between train and validation.
- All CPU performance numbers are engineering smoke records. CUDA BF16, selected fused kernels,
  compile behavior, VRAM, activation-checkpoint tradeoffs, 8K+ hybrid results, and coding effect
  sizes remain MF-050/MF-063 evidence.

## 2026-08-18 — Pre-M10 correctness and release audit

- A bounded local cache with capacity `W` exposes at most `W-1` historical entries to an incoming
  chunk. For single-token decode this is `W-1` old keys plus the current key, exactly matching the
  full-forward window definition; multi-token chunks retain the explicit offset-aware mask.
  Absolute cached key starts therefore use `max(0, start_pos - capacity + 1)`. A regression compares
  full forward with token-by-token decode for more than two window wraps and requires both numerical
  and exact-argmax parity.
- The reported unbounded Flex block-mask growth during decode was already prevented by the
  `auto` dispatcher: cached one-token local attention uses SDPA, while FlexAttention remains the
  training and prefill custom-mask path. Tests assert both the selected backend and a stable block-
  mask-cache size during generation.
- Exception-safe ring-cache rollback remains the correctness default. Its clone/copy traffic is a
  named MF-050 profiling target; it will not be weakened or hidden behind a new default until CUDA
  allocation and latency measurements justify an optimized transaction mode and parity tests cover
  it.
- SFT batches now use a deterministic, epoch-dependent permutation whose seed, policy, epoch,
  cursor, and batch count are checkpointed and validated for exact resume. A contract test proves
  that Jinja rendering, runtime chat encoding, and SFT token serialization agree for multi-turn
  conversations and generation prompts.
- V1 constrains every PyTorch backend extra to `>=2.13.0,<2.14`. The lockfile remains authoritative;
  the minor-version ceiling prevents an unnoticed FlexAttention prototype/API change when the lock
  is regenerated.
- `scripts/build_source_archive.py` is the only documented source-archive path. It includes the
  hidden GitHub Actions workflow, rejects unintended large files, and excludes caches, bytecode,
  weights, corpora, local artifacts, and bundled research/reference archives.
- The historical ARC-Easy/HellaSwag/PIQA profile remains useful continuity evidence but is not a
  release-quality chat/coding claim. MF-066 follows `EVALUATION_RELEASE_GATE.md`, including
  validation, reasoning/knowledge, functional code/FIM, instruction/chat, and trained-context
  retrieval tiers with contamination and not-run reporting.

## 2026-08-18 — M10/M11 software implementation boundaries

- MF-069 uses meta-device construction for exact full-preset parameter accounting and a tiny real
  model only for checkpoint/control-path validation. Its memory figures are analytic lower bounds;
  it cannot produce a scale decision. MF-070 accepts only completed CUDA trainer/profiler evidence
  and retains time, cost, failure, validation, and resume fields in the go/no-go record.
- The Transformers adapter is a standalone copy of the frozen graph rather than a false Llama alias
  or dependency in the neural core. Export tests cover Auto classes, tokenizer/template metadata,
  safe tied weights, cached chunks, and native logits/argmax across Edu, Modern, and global NoPE.
- Transformers 5.15 is the pinned adapter API because the model uses `ALL_ATTENTION_FUNCTIONS` and
  `_supports_attention_backend`. Interleaved local/global layers are declared as
  `sliding_attention`/`full_attention`, allowing vLLM's Transformers backend to allocate the correct
  per-layer cache policy. The WSL2 CUDA run remains the compatibility gate.
- vLLM transport, native parity, and code-edit quality are separate results. Plain completion/chat
  clients are tested without tools; tool/function calling and native Win32 vLLM are not inferred.
- MiniFrontier GGUF uses a distinct `minifrontier` architecture. The conversion runner refuses a
  pinned llama.cpp checkout unless converter, GGUF constants, architecture registry, and compute-
  graph support are all present. Orchestration is not the upstream C++ implementation, so MF-073
  remains open.
- Four-bit work begins only from F16/BF16 GGUF and initially uses Q4_K_M. Calibration provenance is
  explicit, and every candidate remains `publish_ready=false` until CLI/server, memory/throughput,
  tokenizer/template, and quality-regression gates pass on the intended Windows CUDA runtime.
