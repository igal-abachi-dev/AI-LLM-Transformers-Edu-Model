# MF-049 activation-checkpointing speed/VRAM benchmark — RTX 2070 Super

Environment: same as `mf050-rtx2070s-profile-matrix.md` (Windows 11, PyTorch 2.13.0+cu130, RTX 2070
SUPER, 8 GB VRAM). Date: 2026-08-21. Real data: `data/shards/train` (2,213,888 real FineWeb-Edu
tokens; see `reports/mf063-50m-gate.md`). Model: `configs/150m-edu.toml` (154,172,160 parameters),
60 updates, BF16, `train/pretrain.py --device cuda`. Raw records: `reports/mf049-*.json` under
`artifacts/mf049-150m-*/run.json`.

Correctness (loss/gradients agree eager vs checkpointed within tolerance) is separately covered by
the new `test_cuda_activation_checkpointing_matches_loss_and_gradients` test, which passes on this
GPU — see `tests/test_training.py`.

## Results

| Batch size | Checkpointing | Tokens/s | Wall time (60 updates) | Peak allocated VRAM | Outcome |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | off | 2746.9 | 22.3s | 4.43 GB | completes |
| 1 | on | 2134.8 | 28.8s | 3.18 GB | completes; loss identical to eager (`7.9403791427612305` both) |
| 4 | off | — | **did not finish** (killed after 400s+ / no output) | GPU pinned at 7.95/8.19 GB, 100% util | thrashing — VRAM demand exceeds the 8GB card, Windows CUDA sysmem fallback makes it impractically slow |
| 4 | on | 2322.0 | 105.7s | 3.53 GB | completes cleanly |
| 8 | off | — | **did not finish** (killed after 480s+) | same thrashing pattern | not usable on this card without checkpointing |

## Findings

1. **At a batch size that already fits (1), checkpointing costs ~22% throughput for ~28% VRAM
   savings** (2746.9 → 2134.8 tok/s; 4.43 GB → 3.18 GB peak allocated). This matches the expected
   compute-for-memory tradeoff — checkpointing recomputes forward activations during backward instead
   of storing them.
2. **The real value on this hardware is at larger batch sizes eager cannot practically run.** Without
   checkpointing, batch_size=4 already exceeds this 8GB card's usable VRAM: the process didn't crash,
   but sat at 7.95–8.19 GB / 100% GPU utilization for 400+ seconds without completing 60 updates
   (consistent with the same Windows sysmem-fallback thrashing documented in the MF-050 8K-context
   finding). With checkpointing, the identical batch_size=4 run completed cleanly in 105.7s at 3.53 GB
   peak — activation checkpointing is what makes this batch size *usable at all* on this card, not
   just cheaper.
3. **150M-edu training without checkpointing is effectively batch_size≤1-bound on this 8GB card** at
   1024-token sequences — a real, concrete constraint for anyone reproducing this project on similar
   consumer hardware, and the reason MF-063's frozen protocol will need to budget for either a small
   batch size, gradient accumulation, or checkpointing (or all three) rather than assuming the
   README's 24GB baseline.

## Scope note

This benchmark used 60-update bounded runs to keep wall-clock time reasonable for a first measurement
pass, not a full training-length comparison. It reuses the real MF-063 gate data rather than synthetic
tokens, so timings reflect genuine forward/backward/optimizer-step cost on real sequences.

## Update (2026-08-25, MF-075): real FP16 is dramatically faster than emulated BF16 here

The BF16 numbers above use emulated BF16 (this GPU has no native BF16 Tensor Cores — see
`reports/mf050-rtx2070s-profile-matrix.md`). After adding real FP16 + GradScaler support
(`tasks/backlog.md` MF-075), the identical 150M-edu / 60-update / real-data benchmark under FP16:

| Batch size | Checkpointing | Tokens/s | Wall time | Peak allocated VRAM | vs. BF16 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | off | **7424.2** | 8.3s | 3.34 GB | **2.7x faster**, 25% less VRAM |
| 1 | on | **5830.5** | 10.5s | 3.19 GB | **2.7x faster** |
| 4 | off | **10967.7** | 22.4s | 6.65 GB | BF16 never completed at this batch/checkpointing combination (thrashed) — FP16 completes cleanly and fast |
| 8 | off | — | killed (near-VRAM-ceiling thrashing, same pattern as BF16 batch≥4) | GPU pinned at 7.85/8.19 GB | still exceeds this 8GB card without checkpointing |

Raw records: `artifacts/mf075-150m-fp16-b1/run.json`, `artifacts/mf075-150m-fp16-checkpointed-b1/run.json`,
`artifacts/mf075-150m-fp16-b4/run.json`.

This is the headline finding from reviewing `future-plan.md`: real Tensor Core FP16 is not a marginal
improvement over emulated BF16 on this Turing card, it roughly triples real training throughput and
meaningfully raises the largest batch size that fits without activation checkpointing (batch_size=4
now fits eager at 6.65GB, where BF16 needed checkpointing just to survive at that size). **FP16 should
be the default recommendation for training on this hardware going forward** — BF16 remains available
for explicit requests or Ampere+ hardware where it is genuinely native.
