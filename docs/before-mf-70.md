# Pre-MF-070 architecture verification: Muon, local window, global NoPE

## Context

The user asked to "plan first" on several architecture items (MTP head, local window,
global NoPE, hybrid sliding KV cache for local layers, Muon) before running MF-070 (the
350M-parameter scale-check). This was prompted by external AI-generated advice recommending
changes, plus a question about whether recent frontier-model convergence (GLM-5.3-Flash,
Qwen3.8-Flash-Next, Muse Glimmer) means techniques frozen out of V1 scope (DeltaNet, MoE,
MLA, hyper-connections) should be reconsidered.

A verification pass against the actual codebase (not the external advice) found the external
advice was wrong or unverified in several places, and — more importantly — found a real flaw
in the project's own prior evidence. This plan is entirely about **closing real evidence
gaps with bounded, real GPU experiments**, reusing the project's own established
bounded-comparison methodology (same real data, same seed, matched everything except the one
variable under test, real validation loss via `evaluate_token_batches` — not just training
loss). No neural-architecture code changes are proposed.

## What verification found (grounds every decision below)

1. **Hybrid sliding/ring KV cache for local layers — already fully built, tested, measured.**
   `src/minifrontier/cache.py` (`LayerKVCache` with `ring=True`, `KVCache.allocate(...,
   bounded_local=True)`), tested in `tests/test_cache.py`, real measured **~3.4x byte
   savings** (MF-050 status note, `reports/mf050-150m-modern-cuda-8k-hybrid-perfonly.json`).
   Known accepted caveats: ring append is 4.4x slower per-op than linear (allocator/kernel
   overhead, not a memory bug); eager FlexAttention on local layers is a separate bottleneck
   that MF-078 already tried to fix (compiling `flex_attention` directly) and confirmed
   **does not work** on this PyTorch 2.13.0+cu130 build (`InductorError:
   LoweringException: SubgraphLoweringException`, reproduced on both CPU and this GPU).
   **No action needed** — this item is done.

2. **Global NoPE exists, is correctness-tested, but has never been quality-tested.**
   `global_position_encoding` config field (`src/minifrontier/config.py`), implemented in
   `attention.py`, tested for parity in `tests/test_modern.py`. Every real training config
   uses `"rope"`; `labs/07_rope_vs_global_nope.py`'s own docstring says it "proves flag
   isolation, not quality." **Real gap — Priority 3 below.**

3. **Muon has zero real GPU evidence.** MF-055/056/057 are marked Done, but MF-057's own
   status note says the comparison was *"a matched-token CPU engineering sweep...without an
   optimizer-quality claim. The post-M10 RTX rerun owns VRAM, variance, and scale
   conclusions."* That rerun has never happened. `scripts/compare_optimizers.py` already
   supports `--device cuda` and already sweeps both optimizers' LRs separately — it just
   captures no VRAM and (as run so far) no validation loss. **Real gap — Priority 1 below.**

4. **MTP confirmed absent from `src/`,** correctly deferred by both `AGENTS.md`'s frozen-V1
   exclusion list and the external advice's own conclusion ("do NOT add it to the core Modern
   path right now"). **No action.**

5. **Critical flaw in the existing local-window evidence.** The project already ran a real
   `local_window=512` vs `local_window=1024` comparison (`tasks/backlog.md`, MF-070 pre-work,
   2026-08-27) on data packed at `sequence_length=1024`
   (`data/shards/mf064-150m-train/metadata.json`). `src/minifrontier/masking.py`'s window
   rule (`key_position >= query_position - window_size + 1`) becomes a **complete no-op**
   whenever `window_size >= sequence_length`, since every already-causal key then
   automatically satisfies it. Because `1024 == 1024`, the "1024" arm was not "a wider local
   window" — it was **full/global attention with no local restriction at all**. The recorded
   "no meaningful difference" (validation CE 5.145 vs 5.147) is still a valid result, but its
   real meaning is "removing the local/global distinction entirely made no measurable
   difference," not "window sizes 512 vs 1024 are equivalent." This means the external
   advice's recommendation to jump straight to `local_window=2048` should **not** be adopted
   uncritically — at `sequence_length=1024` it would be an equally-degenerate no-op arm.
   **Real gap — Priority 2a below.**

6. **Frontier-model reconsideration (GLM-5.3-Flash / Qwen3.8-Flash-Next / Muse Glimmer).**
   GLM-5.3-Flash: 320B total / 18B active MoE, 34 linear-attention + 11 sparse-attention
   layers of 45, plus Manifold-Constrained Hyper-Connections (mHC, from DeepSeek); no
   consumer-hardware path at all. Qwen3.8-Flash-Next: 125B total / 6B active MoE, alternating
   Gated DeltaNet + Qwen Sparse Attention; minimum self-host target is 96-128GB unified memory
   (Mac Studio / DGX Spark / Strix Halo) — still ~12-16x this project's 8GB card. Muse
   Glimmer: dense, Local/Local/Local/Global + GQA + SwiGLU + RoPE-on-local-only — the only one
   of the three actually comparable in scale and design philosophy to MiniFrontier Modern,
   and it already independently validates the current design. **Conclusion: the convergence
   is real, but it is convergence on solving that scale's problems (massive conditional-MoE
   compute, million-token context economics), not evidence of a gap at 150M-1B dense scale.
   The freeze on MoE/DeltaNet/MLA/hyper-connections stands.** No action.

## Scope (user-confirmed)

Run Priorities **1, 2a, and 3**. Priority 2b (a longer-context, `sequence_length=2048`
re-test, which needs a brand-new data-packing pass) is explicitly deferred — optional,
not now, not core.

---

## Priority 1 — Real Muon-vs-AdamW GPU comparison

Fulfills MF-057's own deferred commitment (the "post-M10 RTX rerun"). Decides what
optimizer MF-070's 350M run (and beyond) should use.

- **Config:** `configs/150m-modern.toml`, unmodified.
- **Data:** `data/shards/mf064-150m-train` (train for training arms, `validation` for the
  post-hoc CE/PPL/BPB pass) — same real data as the existing local-window pre-work.
- **Script:** `scripts/compare_optimizers.py` (already supports `--device cuda`, already
  sweeps LRs). Confirmed CLI: `--config --train-shards --output --updates --batch-size
  --seeds --adamw-lrs --muon-lrs --muon-adamw-lr --match-rms-adamw --device`.
- **Arms:** run each LR as its own single-value invocation (not the full sweep in one
  process) so `torch.cuda.max_memory_allocated()`/`reset_peak_memory_stats()` can be read
  cleanly per arm. AdamW `{1e-4, 3e-4, 1e-3}`, Muon `{3e-4, 1e-3, 3e-3}` → 6 arms, single
  seed 42 (matches this project's established single-seed/exploratory convention).
  ```
  python scripts/compare_optimizers.py --config configs/150m-modern.toml \
    --train-shards data/shards/mf064-150m-train/train \
    --output artifacts/mf070-pre-muon-vs-adamw/<arm-name> \
    --updates 5000 --batch-size 2 --seeds 42 --device cuda \
    --adamw-lrs <one-value>          # or --muon-lrs <one-value>
  ```
- **Bounded scale:** `--updates 5000`, `batch_size=2`, `sequence_length=1024` (from the
  shard manifest) → 2,048 tokens/update × 5,000 = **~10.24M tokens/arm**, matching the
  existing local-window pre-work's scale for comparability.
- **VRAM capture:** a small uncommitted driver script (matching this project's own
  precedent — MF-050's "one-off diagnostic, not committed" — under the scratchpad
  directory) that calls `torch.cuda.reset_peak_memory_stats()`, invokes
  `compare_optimizers.main()` for one arm, and prints peak allocated/reserved bytes after.
  No changes to any tracked file.
- **Validation loss:** for each of the 6 saved checkpoints, load via
  `load_training_checkpoint` and run `evaluate_token_batches` (from
  `src/minifrontier/evaluation/validation.py`) against
  `data/shards/mf064-150m-train/validation`, reusing the exact tokenizer-decode-for-
  utf8-bytes recipe already used for the original local-window test and the MF-065
  validations this session. Report cross-entropy, perplexity, bits/byte per arm.
- **Metrics per arm:** optimizer, LR, seed, consumed tokens, final train loss, wall
  seconds, tokens/sec, peak allocated/reserved VRAM, validation CE/PPL/BPB.
- **Time estimate:** 6 × 10.24M tokens ÷ ~4,200 tok/s (Modern, batch=2, FP16, this
  hardware) ≈ **~4.1 hours**.
- **Recording:** new `reports/mf070-pre-muon-vs-adamw.md` (same table + findings structure
  as `reports/mf050-rtx2070s-profile-matrix.md`, with an explicit single-seed/exploratory/
  10M-tokens-is-far-short-of-3B caveat); a new MF-070 "Pre-work" bullet in
  `tasks/backlog.md` in the exact style of the existing 2026-08-27 entry; update MF-057's
  own status note to point at this report as fulfilling its deferred RTX rerun.

---

## Priority 2a — Corrected local-window re-test

Fixes the real flaw in the prior 512-vs-1024 evidence (finding #5 above) by using window
values that are all genuinely `< sequence_length`, so the local/global distinction is
actually exercised in every arm.

- **Config:** scratch copies of `configs/150m-modern.toml` with only `local_window`
  changed (same pattern as the original pre-work and MF-050's manual-attention scratch
  configs) — the tracked preset file stays untouched.
- **Values:** `local_window ∈ {128, 256, 512}` — all strictly less than
  `sequence_length=1024`. 512 is the current frozen default/anchor point; 128 and 256 are
  genuinely more restrictive, giving a real three-point curve.
- **Data:** `data/shards/mf064-150m-train` (train + validation), identical to Priority 1
  and the original pre-work.
- **Bounded scale:** batch=2, FP16, seed 42, fixed LR `3e-4` (150m-modern's baseline — no
  optimizer sweep needed here since the variable under test is architecture, not
  optimizer), ~10.24M tokens/arm × 3 arms ≈ **~30.7M tokens total**.
- **Metrics:** train loss, validation CE/PPL/BPB via the same `evaluate_token_batches`
  recipe, directly comparable against the existing on-record 512-arm numbers (CE 5.145,
  PPL 171.6, BPB 1.7308).
- **Time estimate:** 3 × 10.24M tokens ÷ ~4,200 tok/s ≈ **~2.0 hours**.
- **Recording:** append a dated "Follow-up" sub-bullet to the existing MF-070 "Pre-work
  (2026-08-27)" entry in `tasks/backlog.md`, explicitly stating what the 512-vs-1024 test
  actually measured (full/global attention vs. hybrid, not two window sizes) before
  presenting the corrected 128/256/512 results; full detail in a new
  `reports/mf070-local-window-corrected.md`.

---

## Priority 3 — Global NoPE quality test

Closes the gap `labs/07_rope_vs_global_nope.py`'s own docstring flags ("proves flag
isolation, not quality"). Lower priority per the data > architecture consensus (both the
external advice and the GLM-5.3 finding agree on this), but cheap and the user opted in.

- **Config:** scratch copy of `configs/150m-modern.toml` with only
  `global_position_encoding` changed to `"none"` (valid: `ModelConfig` only permits
  `"none"` when `attention_pattern == "hybrid"`, which 150m-modern already is). `rope` arm
  is the frozen default, unchanged.
- **Data:** `data/shards/mf064-150m-train`, identical recipe.
- **Bounded scale:** 2 arms (rope vs none), batch=2, FP16, seed 42, fixed LR `3e-4`,
  ~10.24M tokens each ≈ **~20.5M tokens total**.
- **Metrics:** train loss plus validation CE/PPL/BPB, same recipe as above.
- **Time estimate:** 2 × 10.24M tokens ÷ ~4,200 tok/s ≈ **~1.3 hours**.
- **Recording:** new `reports/mf070-global-nope-quality.md`; a short MF-070 Pre-work
  bullet noting this closes the quality-untested gap.

---

## Explicitly out of scope (no plan items)

MTP, MoE, DeltaNet/KDA, MLA, Manifold-Constrained Hyper-Connections, and any exotic
residual/activation tricks (value residuals, x0-residual, smear/backout, logit
soft-capping, ReLU², untied embeddings) remain untouched — consistent with `AGENTS.md`'s
frozen-V1 exclusion list, the external advice's own final conclusion, and finding #6
above. Priority 2b (sequence_length=2048 re-pack) is deferred per explicit user decision,
not bundled into this pass.

## Total cost for this plan

11 arms, ~112.6M tokens, **~7.4 hours** of real GPU time across Priorities 1+2a+3, plus
per-arm CUDA-init/validation overhead. All three experiments reuse
`configs/150m-modern.toml` (unmodified for Priority 1, scratch copies for 2a/3) and
`data/shards/mf064-150m-train`, matching seed 42 / batch=2 / FP16 throughout — the
project's own established bounded-comparison discipline.

## Verification

- Each experiment produces a `reports/mf070-*.md` file with real command output (per
  `AGENTS.md`: "never report a training run, benchmark... as successful without command
  output or a persisted run record") and a corresponding `tasks/backlog.md` status-note
  update in the established format.
- No source code changes are needed for these three experiments (all reuse existing
  scripts/config mechanisms), so the full fast test suite (`pytest -m "not slow"`) and
  `ruff check`/`format --check` are not expected to change — but should still be re-run
  once after the uncommitted VRAM-capture driver script exists, purely as a sanity check
  that nothing in the repo was accidentally touched.
- After all three land, the results directly inform MF-070's own actual 350M profiling
  run (optimizer choice from Priority 1; whether to keep `local_window=512` or adjust from
  Priority 2a; whether to recommend enabling NoPE from Priority 3) before that work begins.
