# M10 optional scale checks

M10 does not make the 350M or approximate 500M presets part of V1. It asks whether a bounded
single-GPU experiment has enough learning value to justify its cost.

## CPU/meta preflight

```powershell
uv run --extra cpu python scripts/preflight_scale.py `
  --output reports/m10-scale-preflight-cpu.json `
  --cpu-checkpoint-smoke
```

This command constructs the full presets on PyTorch's meta device, reports exact parameter counts,
computes labeled BF16/AdamW/KV lower bounds, and exercises forward/backward plus checkpoint/resume on
a tiny structural projection. It never allocates scale-sized activations. Every CUDA field and the
scale decision remain `unmeasured`.

The frozen preset counts are:

| Preset | Exact parameters | Meaning |
| --- | ---: | --- |
| `350m-modern` | 332,460,544 | Optional first scale check |
| `500m-modern` | 433,914,112 | Approximate family name; optional stretch only |

Memory estimates omit activations, kernels, CUDA context, fragmentation, compilation workspaces, and
allocator behavior. They are lower bounds, not capacity promises.

## RTX measurement

Run the existing trainer and profiler against the 350M preset and immutable shards. Start with the
smallest batch/sequence/update budget and activation checkpointing; an OOM is recorded, not hidden.

```powershell
uv run --extra cu130 python train/pretrain.py --config configs/350m-modern.toml `
  --train-shards data/shards/train --output artifacts/m10-350m `
  --device cuda --precision bfloat16 --updates 20 --warmup-updates 2 `
  --batch-size 1 --accumulation-steps 1 --checkpoint-interval 10 `
  --activation-checkpointing

uv run --extra cu130 python scripts/profile_model.py `
  --config configs/350m-modern.toml --output reports/m10-350m-profile.json `
  --device cuda --precision bfloat16 --prompt-length 512 --decode-tokens 32
```

After recording validation loss before/after and proving checkpoint resume, assemble the measured
record and decide:

```powershell
uv run --extra cu130 python scripts/assemble_scale_measurement.py --help
uv run --extra cu130 python scripts/decide_scale.py --help
```

The decision tool refuses CPU, failed, incomplete, or `unmeasured` evidence. It records VRAM,
throughput, elapsed/projected time, projected cost, OOM/non-finite failures, validation behavior,
checkpointing, and expected learning value. A 500M run is forbidden unless that record says `go`.

