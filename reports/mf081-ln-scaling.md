# MF-081: LayerNorm scaling bounded ablation (3 arms)

Real, matched-token, single-seed comparison of LayerNorm scaling
(`layer_norm_scaling`, `1/sqrt(layer_index+1)` multiply on the attention-input
and FFN-input norm outputs, per "The Curse of Depth in LLMs",
arXiv:2502.05795) against the existing init-time `residual_std_damping`
(`residual_std = resolved_init_std / sqrt(2*n_layers)`) already default-on in
`model.py`. Per this task's own acceptance criterion, the point is checking
whether the two residual-stream-growth-damping mechanisms compound or fight
when combined, not just whether LN scaling alone helps.

## Setup

- Config: `configs/150m-modern.toml` (baseline) and two scratch variants
  (`configs/scratch-mf081-ln-scaling-alone.toml`,
  `configs/scratch-mf081-ln-scaling-both.toml`), 150m-modern architecture
  otherwise unchanged.
- Data: `data/shards/mf064-150m-train` (real FineWeb-Edu, 16k tokenizer).
- Training: `train/pretrain.py --updates 5000 --batch-size 2 --seed 42
  --device cuda` (~10.24M train tokens/arm), RTX 2070 Super, FP16.
- Validation: `data/shards/mf064-150m-train/validation`, 435,798 predicted
  tokens / 1,868,949 UTF-8 bytes, evaluated via
  `scripts/eval_checkpoint.py` (new -- see status note below) against each
  arm's `final/` checkpoint.
- Noise floor reference (MF-088, same architecture/data/scale): same-config
  seed-to-seed CE/PPL/BPB spread is real but small; the honest floor is
  "≥0.046%, plausibly ~0.15-0.2%" (`reports/mf088-seed-variance.md`).

## Results

| Arm | `layer_norm_scaling` | `residual_std_damping` | Cross-entropy | Perplexity | Bits/byte |
|---|---|---|---|---|---|
| baseline | off | on (default) | 5.154917 | 173.281 | 1.734139 |
| alone | on | **off** | 5.301092 | 200.556 | 1.783313 |
| both | on | on | 5.136677 | 170.149 | 1.728003 |

Deltas vs. baseline:

| Arm | ΔCE | ΔPPL | ΔBPB |
|---|---|---|---|
| alone | **+2.836% (worse)** | +15.74% (worse) | +2.836% (worse) |
| both | -0.354% (better) | -1.808% (better) | -0.354% (better) |

## Reading

- **LN scaling is not a drop-in replacement for the existing init-time
  damping.** Turning off `residual_std_damping` to add LN scaling in its
  place (`alone`) is a real, large regression (+2.8% CE, +15.7% PPL) --
  far above the ~0.2% noise ceiling, not a single-seed fluke. The two
  mechanisms are not interchangeable at this scale; whatever
  `residual_std_damping` is doing at init time is doing real work that a
  purely-forward-pass `1/sqrt(layer_index+1)` multiply does not replace.
- **LN scaling *added alongside* the existing damping (`both`, matching the
  frozen default's `residual_std_damping=True`) gives a small improvement**
  (-0.354% CE/BPB, -1.8% PPL) -- roughly 2-8x the estimated noise floor, so
  plausibly real, but this is a single seed at a ~10M-token bounded budget,
  not a second-seed-confirmed result.
- **Net effect on the acceptance criterion**: the two mechanisms do not
  compound destructively when both are on (arm `both` beats baseline, not
  just arm `alone`) -- the real risk this task's acceptance criterion asked
  about (silently combining two residual-damping mechanisms) did not
  materialize when both are actually enabled together; it only shows up if
  the older mechanism is turned off to make room for the newer one.
- `layer_norm_scaling` stays off by default (`ModelConfig.layer_norm_scaling:
  bool = False`) pending a decision on whether a single-seed ~0.35% CE gain
  clears the bar for adoption into a real preset, and whether a second seed
  is worth the GPU time to confirm it first.

## Status-note fix found along the way

Evaluating these already-trained checkpoints surfaced a real, general
`load_training_checkpoint` bug (`src/minifrontier/checkpoint.py`): it compared
a checkpoint's raw `config.json` dict against a freshly-built model's
`ModelConfig.to_dict()` directly, so a checkpoint saved before a later
additive `ModelConfig` field existed (here, `rope_fraction`, added by
[[MF-107]] between the `both` and `gqa-2kv` arms finishing) could never be
reloaded again -- the missing key made the raw dicts disagree even though the
model is semantically identical once the new field's default is applied.
Fixed: the saved dict is now round-tripped through `ModelConfig(**saved_config)`
before comparison, so both sides go through the same default-filling
normalization. New regression test:
`tests/test_checkpoint.py::test_training_checkpoint_loads_after_a_new_modelconfig_field_is_added`.
Full fast suite: 337 passed (up from 334), `ruff check`/`format --check` clean.

## Limitations

- Single seed (42), single ~10.24M-token bounded budget -- not a
  second-seed-confirmed result for the `both` arm's improvement.
- Held-out validation quality only; no lm-eval-task or long-context
  evaluation run for this task.
