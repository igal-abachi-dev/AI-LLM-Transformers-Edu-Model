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

## 2026-09-04 — Real RTX 2070 Super evidence: optimizer, position encoding, local window, and compile behavior

- Muon (first-party `torch.optim.Muon`) beats AdamW on quality *per token* at 150m-modern scale
  (val PPL 121.9 vs 169.0, ~10.23M tokens, seed 42) but is ~2.3x slower in wall-clock throughput.
  At *matched wall-clock time* — the real constraint on this project's single 8GB card — AdamW
  wins decisively: real PPL 90.38 (23.94M tokens in ~5,600s) vs Muon's best-case hybrid
  configuration (FP32 Newton-Schulz + `ns_steps=3` + Muon's best known LR, 1e-2) at real PPL 95.86
  for the same wall-clock budget. **AdamW is the optimizer for real training runs on this
  hardware**, including MF-070's 350M run. Muon remains an available experiment with lower peak
  VRAM (5.08GB vs 5.63GB) but no time-matched quality advantage found. See
  `reports/mf070-pre-muon-vs-adamw.md` and `reports/mf070-muon-followup.md`.
- `torch.optim.Muon`'s internal Newton-Schulz iteration is hardcoded to `bfloat16` with no public
  override (confirmed by reading the installed `torch.optim._muon` source directly, not from
  documentation). This RTX 2070 Super (Turing) has no native BF16 tensor cores; emulated BF16
  measurably costs real per-iteration time inside Muon's optimizer step (fitted: ~0.121s/iteration
  in bf16 vs ~0.084s/iteration in fp32, from three real measurements). If Muon is ever used on this
  hardware, a monkeypatch to FP32 gives a real ~21% speedup with matching quality (val CE 4.800 vs
  4.803).
- `torch.compile` does not reliably help on this PyTorch 2.13.0+cu130/Windows build and has two
  independent, real failure modes: compiling `flex_attention` directly fails to lower
  (`InductorError: LoweringException: SubgraphLoweringException`, reproduced on CPU and CUDA —
  MF-078 — independently reconfirmed after ruling out a missing-`triton` explanation); compiling
  `torch.optim.Muon.step()` runs without error but silently corrupts training quality (measured
  perplexity up to ~3.2x worse than the uncompiled control, with inconsistent severity across
  configs) while giving zero real speedup. Do not enable `torch.compile` on either path on this
  build.
- `local_window` size (128/256/512, all genuinely restrictive relative to `sequence_length=1024`)
  has no measurable effect on training quality at this scale (~10.23M tokens/arm; val PPL spans
  only 170.5-171.6, within noise). The frozen default (512) is not starving local layers. A
  longer-context (`sequence_length=2048`) re-test was considered and skipped: the most restrictive
  tested ratio (128:1024, 8:1) already showed nothing, and 512:2048 (4:1) is less restrictive, so a
  repeat is expected to reconfirm rather than reveal anything new. See
  `reports/mf070-local-window-corrected.md`.
- Global NoPE (RoPE on local layers, no positional encoding on the one global layer) does not
  outperform RoPE-everywhere, in-distribution (val PPL 171.6 vs 173.8) or under 2x-length
  extrapolation (both checkpoints improve slightly at 2x length; RoPE stays ahead at both segments;
  the two checkpoints' early-to-late deltas differ by only ~0.007 nats, within single-seed noise).
  RoPE remains the correct default; NoPE stays an available, off-by-default experiment. See
  `reports/mf070-global-nope-quality.md` and `reports/mf070-nope-long-context-extrapolation.md`.
- Reconsidered whether recent frontier-model convergence (GLM-5.3-Flash's MoE + hyper-connections,
  Qwen3.8-Flash-Next's Gated DeltaNet + MoE, Nanbeige 4.2's looped-transformer weight sharing) means
  techniques frozen out of V1 scope should be revisited. Conclusion: no. GLM-5.3-Flash and
  Qwen3.8-Flash-Next solve massive-MoE-scale and multi-hundred-GB-deployment problems irrelevant at
  150M-1B dense scale on one 8GB card. Recurrent-depth/looped-transformer weight sharing (Nanbeige
  4.2, Mixture-of-Recursions, arXiv:2507.10524) trades wall-clock compute for parameter-count
  savings (Nanbeige's own paper: ~75% token efficiency for a 2x-pass loop) — the wrong trade for a
  project that is compute/wall-clock-bound, not parameter-bound; not adopted. The freeze on
  MoE/DeltaNet/MLA/hyper-connections/recurrent-depth stands.

## 2026-09-06 — External code review: verified defects fixed, tokenizer vocabulary reopened, larger proposals bounded

An external line-by-line review of `src/minifrontier/` (config.py, model.py, attention.py,
cache.py, layers.py, rope.py, loss.py, masking.py, training.py, muon.py, mtp.py, precision.py,
data.py, sft.py, tokenizer.py, and the six preset TOMLs) reported no architectural bugs in what
this project has already shipped, plus a set of real gaps and a much larger set of proposed
additions. Per this project's own established discipline (do not accept external review at face
value — see the 2026-09-02/03 Muon/local-window/NoPE verification above), every concrete claim was
independently re-checked against the actual source before any action.

- **Five real defects, confirmed and fixed** (MF-080, Done): non-finite gradients under BF16/FP32
  were applied with no guard (`GradScaler`'s own inf/nan skip only exists under FP16); MTP head
  gradients were excluded from `clip_grad_norm_` since they live outside `model.parameters()`;
  RoPE's rotation multiply ran at the model's running precision (BF16/FP16) rather than FP32,
  losing accuracy in the cos/sin tables that every Q/K in every layer depends on; the FlexAttention
  block-mask cache had no size bound; `MTPHeads`' init std was a hardcoded constant that would
  silently stop matching `ModelConfig.resolved_init_std` once a config set an explicit `init_std`.
  All five fixed with new regression tests (215 passed, up from 208).
- **Tokenizer vocabulary reopened** (2026-09-06, user-approved): 16,384 → 32,768 tokens, plus
  digit-splitting pre-tokenization (Llama 3/Qwen-style, vs. the current GPT-2-style regex that
  merges digit runs into single tokens) — see `AGENTS.md`'s Frozen V1 section for the updated
  line. Grounded in Tao et al. (arXiv:2407.13623): optimal vocabulary size scales with model size,
  and most LLMs (including this project's original 16,384 choice) under-provision it. This is a
  frozen-artifact-breaking decision, explicitly accepted: every checkpoint trained under the
  16,384 tokenizer (MF-063, MF-064/065, the tagged `v0.1.0` release) is incompatible with the new
  tokenizer and is not retroactively upgraded. MF-087 owns the actual retrain.
- **The frozen 3B-token release target was reconsidered and kept unchanged** (2026-09-06,
  user-approved): the review argued for 30-70B tokens minimum, citing SmolLM2/IMU-1 precedents
  using 100-800x more tokens than this project's target. At this hardware's measured ~4,200 tok/s
  (150m-modern), 30B tokens alone is roughly 82 days of continuous GPU time per model size — a
  multi-month project-timeline decision, not a code change. Explicitly kept at 3B; V1 remains an
  honestly-labeled pipeline-validation scale rather than a benchmark-competitive one.
- **A large set of architecture/recipe proposals were bounded into backlog tasks rather than
  adopted directly** (MF-081 through MF-086): z-loss, LayerNorm scaling, value residuals, and
  split local/global RoPE theta (MF-081); per-head gated attention or another attention-sink fix
  for the ring-cache's windowed local layers (MF-082, blocked on a long-context eval that does not
  yet exist); a WSD learning-rate schedule and cautious weight decay (MF-083); chunked
  cross-entropy to reduce logits memory (MF-084 — the review's other suggestion, Cut
  Cross-Entropy/arXiv:2411.09009, needs custom Triton kernels and is not adopted, since
  `AGENTS.md` explicitly excludes those from V1); document-boundary intra-document attention
  masking for packed shards (MF-085); and a broader evaluation harness — BPB, long-context
  retrieval, BLiMP, additional lm-eval tasks, checkpoint EMA (MF-086). None of these are adopted
  yet; each needs its own real, bounded, matched-token test on this project's own model/data/
  hardware before becoming a default, per this project's established practice — a cited paper's
  own numbers (IMU-1's ablations, in particular) are a real prior, not a substitute for that test.
- **One recommendation was checked against existing evidence and rejected**: "promote Muon to
  default," based on IMU-1's iteration/token-matched NorMuon-vs-AdamW result. This project already
  ran the more relevant, decisive test (`reports/mf070-muon-followup.md`): at *wall-clock-matched*
  budgets, AdamW beats every tested Muon configuration on this hardware. Muon stays an experiment,
  not the default.
- **Tokenizer vocabulary reverted to 16,384** (2026-09-08, user-approved): the 2026-09-06 raise to
  32,768 (above) is undone. Real evidence accumulated across three tasks, in order:
  (1) **MF-087's own real comparison** (`reports/mf087-tokenizer-quality.md`) found the opposite of
  the predicted direction — 32k scored *worse* held-out bits-per-byte than 16k at matched
  wall-clock time (1.8039 → 1.8212, +0.956%), with two unresolved confounds (digit-splitting,
  single-seed noise). (2) **MF-088's seed-variance calibration** (`reports/mf088-seed-variance.md`)
  measured this project's first real same-config noise floor (+0.046% relative CE/BPB, seed 42 vs
  43) and showed the 0.956% delta is ~20x that — a real, reproducible effect, not noise.
  (3) **MF-090's investigation** closed the remaining confounds: a no-GPU fertility triage
  (`artifacts/mf090-tokenizer-fertility/fertility.json`) found the shipped 32k tokenizer was
  actually *more* fertility-efficient than 16k (+5.4%), ruling out raw tokenization inefficiency or
  digit-splitting as the cause; a periodic-validation rerun
  (`reports/mf090-periodic-validation-curve.md`) showed 32k starting *ahead* on BPB early in
  training, crossing over, and 16k's lead *widening* for the rest of the bounded budget — the
  opposite of the pattern a short-token-budget recovery would produce, and cross-entropy diverged
  monotonically throughout. Taken together: the regression is real (not noise), not explained by
  tokenization efficiency or digit-splitting, and does not resolve itself within the tested
  ~7-8M-token budget by continuing to train the same way. This does not prove 32k would still lose
  at the full compute-optimal 3B-token target Tao et al.'s theory was calibrated for (this bounded
  test covers only ~0.26% of that budget) — but absent any evidence in 32k's favor at any tested
  scale, and given a real, reproducible cost measured three independent times, the frozen
  vocabulary reverts to 16,384 rather than carrying an unproven bet into the real release run.
  **Digit-splitting is reverted alongside it, not combined with 16,384 instead**: "16k +
  digit-split" was never trained or tested in any form — adopting it now would repeat exactly the
  "merged in blind" mistake MF-090 existed to prevent. `train_byte_bpe`'s `digit_split` parameter
  (added during this investigation) remains available, off by default, for a future test that
  isolates digit-splitting's actual purpose (arithmetic capability) rather than only measuring its
  fertility cost. Migration cost is low: no real production-scale checkpoint was ever trained under
  32k (only the bounded ~7-8M-token comparison arms above); the original 16,384-tokenizer
  checkpoints this project already has (MF-063/064/065, `v0.1.0`) remain exactly what they were.

## 2026-09-08 — MF-081/082/083/106 architecture surface: Modern-only scope split, and one direct adoption

- **Every architecture-level item introduced by MF-081/082's bounded ablations is Modern-only**
  (user-directed): LayerNorm scaling, value residuals, split local/global RoPE theta, GQA-ratio
  changes, and per-head gated attention. Edu must stay exactly the classic architecture
  (`AGENTS.md`'s own Frozen V1 list) so it remains simple enough to fully explain to a beginner
  regardless of what Modern's bounded ablations end up adopting. `ModelConfig.__post_init__` now
  rejects `layer_norm_scaling`/`value_residual`/`gated_attention` on `preset="edu"`; GQA-ratio
  changes and a non-default `global_rope_theta` were already structurally impossible on Edu via its
  existing MHA-only/full-attention-only guards. Mirrored into the HF export adapter's own Edu guard.
  MF-083's two training-recipe items split further, since neither is a `ModelConfig` field and
  `TrainingConfig` has no preset awareness to validate against: WSD is an operational safeguard
  against an interrupted multi-day run, not a quality technique, so it applies to *both* Edu's and
  Modern's real release runs (user-confirmed); cautious weight decay is a real quality technique
  gated on its own bounded test, so it stays Modern-only like the architecture items above.
- **Split local/global RoPE theta (MF-081) is adopted directly** (no bounded ablation needed, per
  this project's own "zero cost, well-attested, can't plausibly hurt" rule already used for z-loss):
  `ModelConfig.global_rope_theta`, defaulting to `None` (= same as `rope_theta`, so every existing
  config/checkpoint/test is unaffected). No frozen preset sets a non-default value; the mechanism
  exists and is tested, not yet applied to any real release config.
- **SwiGLU clamping (MF-106) is adopted directly**, same rule, and — unlike the MF-081/082 items
  above — is *not* Modern-only: it is a numerical-safety net shared by both frozen presets, not a
  competing architectural identity. Raised from a real, verified read of the DeepSeek-V4 technical
  report (arXiv 2606.19348, confirmed real via `WebSearch` + a direct primary-source fetch of
  `arxiv.org/html/2606.19348v1`, not taken on faith from secondary summaries — see MF-106/MF-107 in
  `tasks/backlog.md` for the full verification trail, including a real correction: the paper's own
  cited "64 dimensions" for its partial-RoPE feature is DeepSeek's own absolute dimension count for
  a differently-shaped attention mechanism, not a literal "50%" this project could copy). Section
  4.2.3, "Mitigating Training Instability," confirmed present with a "SwiGLU Clamping" subsection;
  its exact numeric clamp values could not be confirmed from the primary source text itself (only a
  secondary, unverified source claims `[-10, 10]`/cap-10) — `ModelConfig.swiglu_clamp: float | None
  = None` therefore ships as a mechanism with no project-chosen default value, not a citation-grade
  number. Real, not hypothetical, for this project specifically: DeepSeek trains in BF16/FP8 (wide
  dynamic range); this project's own documented precision policy trains in FP16 with gradient
  scaling on non-native-BF16 hardware (`AGENTS.md`, `reports/mf049-rtx2070s-checkpointing-benchmark.md`),
  where an activation spike is a genuinely bigger risk, not a smaller one.
- **DeepSeek's own real attention-sink mechanism was found and recorded, but the fix already chosen
  and built for MF-082 (per-head gated attention) was not replaced.** V4's real mechanism (confirmed
  from the primary source): one learnable scalar `z'_h` per head, added as an `Exp(z'_h)` term
  inside the softmax denominator, letting a head's total attention weight sum to less than one. This
  is the same shape as the "gpt-oss-style learned per-head sink logits" alternative MF-082's own
  task description already named but did not choose — recorded as a precisely-specified fallback
  candidate for later, not acted on now, since swapping out already-implemented, already-tested code
  without evidence it underperforms would repeat the mistake this project's own bounded-comparison
  discipline exists to prevent.
- **DeepSeek V4.1 Flash** (a separate, unrelated announcement in the same feedback batch) was
  checked and found to carry no technical content: a two-day internal API beta with a speed
  benchmark and a marketing claim of a new multimodal architecture, no technical report, weights,
  config, or architecture disclosure. Reviewed, not acted on.

## 2026-09-09 — MF-081/082: LayerNorm scaling and gated attention adopted into every Modern preset

- **`layer_norm_scaling = true` and `gated_attention = true` are now the default for every frozen
  Modern preset** (`configs/50m-modern.toml`, `150m-modern.toml`, `350m-modern.toml`,
  `500m-modern.toml`; `residual_std_damping` stays at its existing default `true`), user-directed
  after the real bounded-ablation results below. Both remain structurally impossible on Edu
  (`ModelConfig.__post_init__`'s existing guard); no change to Edu's frozen architecture.
- **LayerNorm scaling**: the real 3-arm bounded test (`reports/mf081-ln-scaling.md`) found LN
  scaling *replacing* the existing init-time `residual_std_damping` is a clear regression (+2.84%
  CE) — not adopted that way. LN scaling *added alongside* the existing damping (the arm actually
  adopted here) showed a small improvement (-0.35% CE/BPB, -1.81% PPL), 2-8x the estimated noise
  floor (~0.046-0.2%, `reports/mf088-seed-variance.md`) but from a single seed at a ~10.24M-token
  bounded budget — plausibly real, not confirmed by a second seed. Adopted as a pragmatic call given
  the downside case (replacing the existing damping) is now known and explicitly avoided.
- **Gated attention** (per-head sigmoid gate on the SDPA output before `out_proj`, Qiu et al.
  arXiv:2505.06708, Qwen3-Next-style — the fix MF-082 chose for the ring-cache attention-sink
  degradation `reports/mf086-needle-haystack.json` confirmed): the real bounded arm
  (`reports/mf082-gated-attention-quality.md`) is the clearest quality signal of any arm run this
  pass, -0.67% CE / -3.37% PPL / -0.67% BPB, well above the noise floor. **Adopted for this
  validation-quality result alone** — the companion needle-haystack re-eval (this task's actual
  sink-fix acceptance test) came back inconclusive, not negative: both the baseline and gated arms
  scored 0.0 retrieval at every context length including inside the local window, because a
  5,000-update (~10.24M-token) bounded checkpoint is ~100x short of the reference budget
  (`reports/mf086-needle-haystack.json`'s real 1B-token MF-065 release, which does retrieve within
  its local window) where the retrieval capability has actually emerged. Whether gated attention
  specifically fixes sink eviction remains an open question, explicitly deferred: a conclusive
  re-test needs a real ~1B-token `gated_attention=true` retrain (3-4 days at this project's measured
  throughput), which is real GPU-time competition with MF-070's own 350M run and is deferred until
  after it, not run now. The formula itself (sigmoid gate on the attention output) is not claimed to
  be the best possible mechanism — DeepSeek-V4's real, more surgical alternative (a learned per-head
  scalar inside the softmax denominator, letting attention mass sum to less than one, see the
  2026-09-08 entry above) remains recorded and available; switching to it is deferred until there is
  real evidence the current sigmoid formula underperforms specifically at the sink-eviction task,
  not swapped speculatively now.
- **Real parameter-count effect, computed via `exact_parameter_count`, not hand-calculated**:
  `gated_attention`'s `gate_proj: Linear(d_model, n_heads, bias=True)` per layer is the only one of
  the two adopted items that adds parameters. 50m-modern: 47,915,376 (was 47,730,816, +0.387%);
  150m-modern: 138,630,640 (was 138,446,080, +0.133%); 350m-modern: 332,919,744 (was 332,460,544,
  +0.138%); 500m-modern: 444,244,380 (was 443,621,760, +0.140%). `layer_norm_scaling` adds zero
  parameters (a fixed, non-learned multiply).
- **GQA ratio stays at each preset's existing default (3:1) — not changed.** The real 2-arm sweep
  (`reports/mf081-gqa-sweep.md`) found 6:1 statistically indistinguishable from 3:1 at 150M scale
  (-0.015% CE, inside the noise floor) and 12:1/MQA a small, plausibly-real regression (+0.24% CE).
  Nothing between 6:1 and 12:1 was tested, this is single-seed at 150M scale on a ~10.24M-token
  budget (not the 350M scale or 3B-token release budget this decision will actually govern), and
  6:1 showed no quality upside over 3:1 to justify switching on its own — only a KV-cache memory
  argument, which this project has not needed. Kept at 3:1 pending either a stronger reason to move
  or a real test at 350M scale.

## 2026-09-09 — MF-105: MTP enabled for the real 350M/release run

- **`--mtp-extra-heads 1 --mtp-loss-weight 0.3` will be used for the real 350M/release training
  run**, user-confirmed, decided ahead of MF-105's own original framing ("wait until MF-070's run is
  being planned, not before") — grouped instead with this session's other pre-MF-070 recipe
  decisions (LN scaling, gated attention, GQA ratio) rather than deferred further.
- **Why**: the original "MTP stays off by default" call (`reports/mf070-mtp-quality.md`) weighed
  only a quality-only tradeoff (−5.3% training wall-clock for +0.74% PPL — not worth it alone).
  MF-093's later, separate result changed the calculus: the same trained MTP heads, repurposed for
  self-speculative decoding, give a real, measured 1.21x greedy-decode speedup at 45.3% draft
  acceptance (`reports/mf093-speculative-decoding.md`) — a benefit the original decision never had
  available to weigh. The training-time cost is paid once during the (already multi-day) run; the
  drafting capability then persists on the checkpoint for every future greedy-decode use
  indefinitely. Architectural risk is low by construction: MTP heads are carved out to never touch
  `ModelConfig` or `MiniFrontier`'s `state_dict()` (the frozen MTP carve-out, `AGENTS.md`), so this
  does not complicate the release checkpoint format or compatibility.
- **Caveats accepted, not overlooked**: the 1.21x/45.3% figures are single-seed, measured at 150M
  scale, not yet confirmed at 350M; the speedup is greedy-only today — `chat.py`'s real default
  (`temperature=0.8`) will not benefit until a materially more complex probability-ratio acceptance
  rule for non-greedy sampling exists, which is unscoped future work, not part of this decision.
  Edu is unaffected either way (never trained with MTP heads).
- **Does not change any code-level default.** `TrainingConfig.mtp_extra_heads` stays `0` in the
  library itself — this is a decision about the real release run's actual invocation command, the
  same category of recipe decision as WSD's decay fraction, not a change to MTP's frozen
  off-by-default carve-out.
- **Follow-up, same day: the speedup was undeliverable until [[MF-109]] closed a real gap.**
  `export_release`/`load_release` never learned about MTP heads when MF-093 added `mtp_heads`
  support to the training-checkpoint save/load path — a release exported from an MTP-trained
  checkpoint would have silently dropped the draft heads, and `scripts/sample.py` had no path to
  use them regardless. `scripts/export.py`/`checkpoint.py`/`release.py`/`chat.py`/`scripts/sample.py`
  now carry MTP heads end to end, and self-speculative decoding is on by default in
  `scripts/sample.py` whenever a release has trained heads and the request is genuinely greedy —
  structurally Modern-only (Edu never has heads to export) rather than via a preset check. Full
  detail in `tasks/backlog.md`'s `MF-109` entry.
