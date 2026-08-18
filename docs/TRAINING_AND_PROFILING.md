# Training and profiling

`train/pretrain.py` is the canonical single-process trainer. It consumes immutable shard
directories produced by `scripts/prepare_data.py`; it never consumes a live network iterator.
The preparation CLI accepts either a provenance-complete `--manifest` or the pinned
`--source fineweb-edu` stream directly. It writes separate hashed `.npy` token/count arrays so each
worker can memory-map and cache its current shard instead of reopening an NPZ container per sample.

The loop uses AdamW with matrix/embedding decay groups, update-indexed warmup plus cosine decay,
global gradient clipping, exact target-token-weighted accumulation, optional whole-block activation
checkpointing, and `auto|float32|bfloat16` precision. Checkpoints contain safe model weights plus a
trusted-local optimizer/scheduler/RNG/data-cursor file. Resume rejects model or training-config
changes rather than silently changing the experiment.
Training order is a seed-derived shard permutation plus a bounded per-shard row permutation. The
epoch, shard cursor, row cursor, seed, and shuffle policy are checkpointed and validated on resume.

Typical bounded CPU validation:

```bat
uv run --extra cpu python train/pretrain.py --config configs/50m-modern.toml ^
  --train-shards artifacts/data/train --output artifacts/run --device cpu ^
  --attention-impl sdpa --precision float32 --updates 2 --warmup-updates 0
```

Modern CPU training must select `sdpa`; the pinned PyTorch FlexAttention implementation supports
CPU inference/reference checks but not CPU backward. On CUDA, the Modern `auto` policy selects
FlexAttention for local layers and SDPA for global layers.

`scripts/profile_model.py` records a labeled, single-stream prefill/decode result. Context/window
overrides require `--performance-only-override` and never imply trained long-context quality.
Run the full 50M/150M BF16, eager/compile/checkpointing, VRAM, and 8K+ matrix on the home GPU under
MF-050/MF-063; CPU numbers are useful only for integration regression checks.
Modern profiling uses bounded local ring storage by default and the full-history reference with
`--full-history-local-cache`. Single-token local decode uses SDPA over the already bounded cache;
FlexAttention remains the local training/prefill custom-mask path.
