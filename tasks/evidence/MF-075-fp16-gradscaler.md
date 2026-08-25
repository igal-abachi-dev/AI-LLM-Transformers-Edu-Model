# MF-075 — real FP16 + GradScaler evidence

Environment: Windows 11, Python 3.12.10, PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 2070 SUPER
(Turing, CC 7.5, 8GB VRAM). Date: 2026-08-25.

## Why

Reviewing `future-plan.md` (a compiled set of external AI reviews of this project) surfaced a
concrete, code-verifiable finding: `src/minifrontier/precision.py` resolved `"auto"`/`"bfloat16"`
via `torch.cuda.is_bf16_supported()` with PyTorch's default `including_emulation=True`, which
reports `True` on this Turing card even though Turing has no native BF16 Tensor Cores (only Ampere+
does). Every prior GPU evidence run in this project (MF-046/049/050/063) therefore measured emulated,
not accelerated, BF16 — independently corroborated by `torch.compile`'s own "does not support
bfloat16 compilation natively" warning and by FP32 outmeasuring BF16 in the MF-050 profiling matrix.

## What changed

- `src/minifrontier/precision.py`: `Precision` extended to include `"float16"`;
  `resolve_precision`'s `"auto"` mode now prefers native BF16
  (`is_bf16_supported(including_emulation=False)`) → FP16 on any other CUDA device → FP32, instead of
  PyTorch's emulation-inclusive default. Explicit requests (`"bfloat16"`, `"float16"`, `"float32"`)
  are always honored as asked. `"float16"` on CPU falls back to FP32 (PyTorch's CPU FP16 op coverage
  is weak, unlike its CPU BF16 coverage).
- `src/minifrontier/training.py`: `train_updates` now wraps the backward/clip/step sequence with a
  `torch.amp.GradScaler`, constructed with `enabled=policy.needs_grad_scaler` so every call
  (`scale`/`unscale_`/`step`/`update`) is a transparent no-op for BF16/FP32 runs — verified empirically
  against both plain AdamW and the Muon `CombinedOptimizer`. A new `TrainingState.grad_scaler_state`
  field persists the scaler's state; because it is just another `TrainingState` field, it round-trips
  through the *existing* generic JSON-based checkpoint save/resume path with zero changes needed to
  `checkpoint.py`, `train/pretrain.py`, or `train/sft.py`.
- All 7 CLIs that expose `--precision {auto,float32,bfloat16}` extended to include `float16`:
  `scripts/chat.py`, `scripts/eval.py`, `scripts/eval_sft.py`, `scripts/profile_model.py`,
  `scripts/sample.py`, `train/pretrain.py`, `train/sft.py`.
- Six new tests in `tests/test_training.py`: two CPU-only (FP16-on-CPU fallback; `"auto"` prefers
  native BF16 over emulated BF16 on a monkeypatched-Turing-shaped CUDA device, using the exact
  `including_emulation` keyword PyTorch's real signature exposes) and four real-CUDA (`@requires_cuda`,
  matching the existing MF-046/049 pattern): full-vs-cached FP16 logit parity with a tolerance declared
  before measurement (`atol=5e-2`), a training-step sanity check (finite loss, weights actually
  change, `state.completed_updates` advances unconditionally), and a checkpoint/resume test proving
  the scaler's scale factor survives a save/reload rather than silently resetting to PyTorch's
  default.

## Real re-measurement (not just implementation)

Reran a scoped subset of the already-collected MF-049/MF-050 evidence under real FP16:

- `reports/mf049-rtx2070s-checkpointing-benchmark.md` (2026-08-25 update): 150M-edu training,
  60 real updates, real FineWeb-Edu data. FP16 batch_size=1 eager: 7,424 tok/s vs BF16's 2,747 tok/s
  (**2.7x**), using less VRAM (3.34GB vs 4.43GB). FP16 batch_size=4 eager: 10,968 tok/s and completes
  cleanly at 6.65GB, where the identical BF16 configuration never completed (VRAM-thrashing into
  Windows' CUDA sysmem fallback). Raw records: `artifacts/mf075-150m-fp16-*/run.json`.
- `reports/mf050-rtx2070s-profile-matrix.md` (2026-08-25 update): single-stream inference is more
  mixed — FP16 decode beats emulated BF16 but FP32 remains fastest for both prefill and decode at
  this small batch=1 scale, reported as measured rather than assumed. Raw records:
  `reports/mf075-50m-edu-cuda-sdpa-fp16.json`, `reports/mf075-150m-edu-cuda-sdpa-fp16.json`.

**Recommendation carried forward**: FP16 is now the default recommendation for *training* on this
card; BF16 remains available for explicit requests or genuinely native (Ampere+) hardware. Inference
precision needs a separate, larger-scale measurement pass before recommending anything definitively.

## Verification

- Ruff lint and format: clean.
- Fast suite: 191 passed total (187 non-slow + 4 slow), up from 186 before this task — 6 new tests,
  1 pre-existing gap closed (`../train/pretrain.py --help` was already added to `tests/test_clis.py`
  in the prior session).
- All new tests execute for real (not skipped) on this CUDA hardware and pass.
