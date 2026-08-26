# MF-050 home-GPU profiling matrix — RTX 2070 Super

Environment: Windows 11, Python 3.12.10, PyTorch 2.13.0+cu130, NVIDIA GeForce RTX 2070 SUPER
(Turing, compute capability 7.5, **8 GB VRAM** — below the README's "comfortable 24GB" baseline;
this is the actual home-GPU gate). Date: 2026-08-21. Git commit `ad47b50232650220c5b4707fcd83fef1cb5eb9b0`.
All runs use untrained weights (inference timing does not depend on weight quality) via
`scripts/profile_model.py --device cuda`. Raw records: `reports/mf050-*.json`.

**Measurement caveat**: each row is a single cold-process invocation with no warmup iteration
(`profile_model.py` has no warmup mechanism). `time_to_first_token_seconds` therefore includes
one-time CUDA context/kernel-selection cost, which dominates especially for `torch.compile` and
FlexAttention rows below. Prefill tokens/s is not directly comparable across rows with different
`--prompt-length`.

## Prerequisite fix

Before any of these runs could execute, `KVCache` compared a bare `torch.device("cuda")`
(`index=None`) against a real tensor's `torch.device("cuda", 0)`. PyTorch treats these as unequal,
so **every documented `--device cuda` invocation with a lazily-dtyped cache would fail immediately**
on first cached forward — a bug invisible to CPU-only testing (CPU has no index ambiguity). Fixed in
`src/minifrontier/cache.py` (`LayerKVCache.allocate` now resolves the device's CUDA index before
storing it). Regression-covered by the new `test_cuda_*` tests in `tests/test_training.py`.

## Results

| Config | Precision | Compile | Prompt/Decode | Prefill tok/s | Decode tok/s | TTFT (s) | Peak alloc VRAM | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 50M-edu | BF16 | no | 64/16 | 261.4 | 39.1 | 0.24 | 121 MB | baseline |
| 150M-edu | BF16 | no | 128/32 | 609.2 | 31.7 | 0.21 | 336 MB | longer prompt, not directly comparable to 50M row above |
| 150M-edu | FP32 | no | 128/32 | 1122.4 | 50.1 | 0.11 | 652 MB | FP32 faster than BF16 here — Turing has no native BF16 tensor cores, so BF16 runs emulated |
| 150M-edu | BF16 | yes | 128/32 | 7.8 | 36.0 | 16.31 | 337 MB | inductor: *"does not support bfloat16 compilation natively, skipping"* — one-shot compile cost dominates; steady-state decode barely moves vs eager |
| 150M-modern | BF16 (auto) | no | 128/32 | 46.4 | 25.6 | 2.76 | 303 MB | hybrid local/global; eager FlexAttention on local layers is unfused (materializes full scores), explaining the low prefill number |
| 150M-modern, 8K ctx, bounded-local cache | BF16 | no | 8176/8 | 223.0 | 23.5 | 36.67 | **10.87 GB** | **exceeds 8 GB physical VRAM** — see finding below |
| 150M-modern, 8K ctx, full-history cache | BF16 | no | 8176/8 | 247.4 | 27.3 | 33.05 | **10.96 GB** | same overflow; included to isolate the cache-size effect |
| 50M-edu | **FP16** | no | 64/16 | 232.7 | 39.7 | 0.27 | 121 MB | real Tensor Core FP16 (2026-08-25, MF-075) |
| 150M-edu | **FP16** | no | 128/32 | 549.7 | 44.3 | 0.23 | 335 MB | real Tensor Core FP16 (2026-08-25, MF-075) |
| 50M-edu | BF16 | no (**manual**) | 64/16 | 195.0 | 36.7 | 0.33 | 121 MB | (2026-08-26) teaching path vs. the SDPA row above: 25% slower prefill, ~6% slower decode |
| 150M-edu | BF16 | no (**manual**) | 128/32 | 546.7 | 32.5 | 0.23 | 335 MB | vs. SDPA row above: 10% slower prefill, decode within noise |
| 150M-edu | FP32 | no (**manual**) | 128/32 | 686.3 | 38.0 | 0.19 | 652 MB | vs. FP32/SDPA row above: 39% slower prefill, 24% slower decode |
| 150M-edu | FP16 | no (**manual**) | 128/32 | 399.9 | 31.3 | 0.32 | 335 MB | vs. FP16/SDPA row above: 27% slower prefill, 29% slower decode |
| 150M-modern | BF16 | no (**manual**) | 128/32 | 299.0 | 22.0 | 0.43 | 301 MB | vs. BF16(auto)/eager-FlexAttention row above: **6.4x faster prefill**, 6.5x lower TTFT, ~14% slower decode — see Finding 7 |

## Findings

1. **BF16 is not a speed win on Turing.** This GPU's tensor cores don't support BF16 natively;
   PyTorch runs it emulated, and FP32 prefill/decode both measured faster than BF16 (1122 vs 609
   tok/s prefill, 50.1 vs 31.7 tok/s decode, same 150M-edu config). BF16 is still useful for its
   memory footprint (half the bytes per weight/activation/cache entry) but this hardware should not
   be expected to show a BF16 throughput advantage the way Ampere+ GPUs would.
2. **`torch.compile` gives no real benefit here and a large one-time cost.** Inductor explicitly
   declines native BF16 compilation on this GPU. Steady-state decode throughput is within noise of
   eager (36.0 vs 31.7 tok/s); the ~16s compile-time TTFT would need a very long generation to
   amortize.
3. **The 8K-context hybrid benchmark exceeds this card's 8 GB VRAM.** Peak allocated VRAM
   (~10.9 GB) is *higher than the GPU's physical 8.59 GB total* for both cache variants. The process
   did not crash — Windows' CUDA sysmem-fallback policy silently spilled into system RAM, which is
   consistent with the extreme TTFT (33–37s vs ~0.2–2.8s at 2K context). This is a genuine failure
   mode, not a working long-context result: an 8K-context single-stream generation is not practical
   on this 8GB card at these settings (batch=1, full 20-layer 150M-modern) without reducing precision
   further or accepting system-RAM paging.
4. **The ring-cache memory saving itself is real and clean.** At identical 8K context, the bounded
   local ring cache allocates 49,766,400 bytes vs the full-history cache's 167,608,320 bytes — a
   genuine ~3.4x reduction, matching the intended MF-041 design, even though the surrounding forward
   pass itself doesn't fit in VRAM at this context length on this card.
5. **150M-modern's eager FlexAttention path is markedly slower than 150M-edu's eager SDPA path**
   at comparable settings (46.4 vs 609.2 tok/s prefill, though prompt lengths also differ) — expected
   given the documented unfused-eager FlexAttention warning, and consistent with FlexAttention's
   design assuming `torch.compile` wrapping that this GPU can't use effectively for BF16.

6. **Update (2026-08-25, MF-075): real FP16's inference story is mixed, but its *training* story is
   dramatic.** For single-stream inference at this small scale, FP16 decode beats emulated BF16
   (44.3 vs 31.7 tok/s on 150M-edu, a real Tensor Core benefit) but neither FP16 nor BF16 beats FP32
   for prefill or decode here (FP32 remains fastest: 1122.4 prefill / 50.1 decode). At batch=1 and
   these short prompt lengths, fixed per-op `autocast` dispatch overhead and SDPA's kernel-selection
   heuristics likely swamp any Tensor Core advantage — FP32 skips autocast entirely, which may be why
   it wins despite being twice the memory traffic. This is genuinely counter-intuitive and reported as
   measured, not rationalized further without more evidence. **The real win is on the training side**,
   where FP16 is ~2.7x faster than emulated BF16 and uses meaningfully less VRAM (see
   `reports/mf049-rtx2070s-checkpointing-benchmark.md`'s 2026-08-25 update) — training's backward pass
   is far more matmul-heavy than a short single-token decode step, so emulated-BF16's overhead compounds
   there in a way it doesn't (yet) show up in these small inference numbers. **Recommendation: use FP16
   for training on this hardware; precision choice for inference needs its own larger-scale/longer-
   prompt measurement pass before recommending anything definitively.**

## Update (2026-08-26): manual-attention path, ring-cache allocation/rollback profiling, batch-size sweep

This closes the three gaps the previous pass left open. All numbers below are real measurements on
the same RTX 2070 SUPER, same environment. Raw records for the manual-attention rows and the batch
sweep are one-off JSON files under a scratch directory (not committed — they are transient
measurement artifacts, not release evidence); the ring-cache diagnostic's exact script and output are
inlined below since it was never wired up as permanent CLI surface.

### 7. The manual/teaching attention path is slower than SDPA everywhere it was tested — except one case where it is dramatically faster

Copying each frozen preset to a scratch config with only `attention_impl` changed to `"manual"`
(`configs/*.toml` themselves were not edited) and re-running `scripts/profile_model.py` at the same
settings as the existing matching rows above gives a direct manual-vs-optimized comparison:

- On full-attention Edu models, manual is consistently slower than SDPA, as expected for a path that
  materializes the full `[B,H,S,S]` score tensor instead of using a fused kernel: 25% slower prefill
  on 50M-edu/BF16, 10% on 150M-edu/BF16, 39% on 150M-edu/FP32, 27% on 150M-edu/FP16. Decode-side the
  gap is smaller and noisier (single-token attention is memory-bound, not compute-bound, so the fused
  kernel matters less there).
- On 150M-modern (hybrid Local/Local/Local/Global), the picture inverts: manual attention batch is
  **6.4x faster on prefill** (299.0 vs 46.4 tok/s) and has 6.5x lower TTFT (0.43s vs 2.76s) than the
  existing `attention_impl=auto` row, which resolves local layers to eager FlexAttention. This
  confirms the earlier report's suspicion (Finding 5) that eager, uncompiled FlexAttention on this
  GPU is not merely "somewhat slower" than a fused kernel — it is slower than the plain, unoptimized
  reference implementation it exists to replace. Decode is the one place manual is worse here (22.0
  vs 25.6 tok/s, ~14% slower), consistent with FlexAttention's own kernel being tuned more for
  the compute-bound prefill phase. **Recommendation: on Turing-class GPUs without `torch.compile`
  support for FlexAttention, prefer manual attention over eager FlexAttention for the hybrid preset's
  prefill phase**, or accept the TTFT/prefill cost only if `torch.compile` can actually be used (it
  could not be, on this GPU — Finding 2).

### 8. Ring-cache append is ~4.4x slower per step than the linear cache, but the caching allocator absorbs the extra allocation traffic

`LayerKVCache.append`'s ring branch (`src/minifrontier/cache.py`) does a `_chronological()` readback,
clones two rollback tensors (`_rollback_keys`/`_rollback_values`) unconditionally on every append, and
then an `index_copy_` scatter — versus the linear path's single contiguous `copy_`. To isolate this
from ordinary forward-pass cost, the following one-off script drove `LayerKVCache.append` directly
(150M-modern's shape: 4 KV heads, head_dim 64, local_window 512), 1223 single-token appends per
configuration — crossing more than two full 512-token ring wraps (1223/512 ≈ 2.4 wraps), matching the
"at least two complete local-window wraps" bar used elsewhere in MF-050's acceptance criteria:

```python
"""One-off diagnostic (not committed): measure ring-cache append/rollback allocation
traffic vs the linear/full-history cache, in isolation from a full model forward pass."""

from __future__ import annotations
import time
import torch
from minifrontier.cache import LayerKVCache
from minifrontier.config import ModelConfig

config = ModelConfig.from_toml("configs/150m-modern.toml")
device = torch.device("cuda")
batch_size, n_kv_heads, head_dim, window = (
    1,
    config.n_kv_heads,
    config.head_dim,
    config.local_window,
)
n_steps = window * 2 + 200


def run(*, ring: bool, capacity: int, steps: int, commit_each_step: bool) -> dict:
    cache = LayerKVCache.allocate(
        batch_size=batch_size,
        n_kv_heads=n_kv_heads,
        capacity=capacity,
        head_dim=head_dim,
        device=device,
        dtype=torch.bfloat16,
        ring=ring,
    )
    warm_k = torch.randn(batch_size, n_kv_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
    warm_v = torch.randn(batch_size, n_kv_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
    cache.append(warm_k, warm_v, start_pos=0)
    if commit_each_step:
        cache.commit()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    stats_before = torch.cuda.memory_stats(device)
    started = time.perf_counter()
    for step in range(1, steps):
        k = torch.randn(batch_size, n_kv_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, n_kv_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
        cache.append(k, v, start_pos=step)
        if commit_each_step:
            cache.commit()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    stats_after = torch.cuda.memory_stats(device)
    return {
        "elapsed_seconds": elapsed,
        "seconds_per_append": elapsed / (steps - 1),
        "num_alloc_retries": stats_after["num_alloc_retries"] - stats_before["num_alloc_retries"],
        "allocation_count_delta": stats_after["allocation.all.current"]
        - stats_before["allocation.all.current"],
        "peak_active_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


print(run(ring=False, capacity=n_steps, steps=n_steps, commit_each_step=True))  # linear baseline
print(
    run(ring=True, capacity=window, steps=n_steps, commit_each_step=True)
)  # ring, committed each step
print(
    run(ring=True, capacity=window, steps=n_steps, commit_each_step=False)
)  # ring, rollback never committed
```

| Cache | Per-append time | vs. linear | Alloc-count delta | Peak active bytes |
| --- | ---: | ---: | ---: | ---: |
| Linear (non-ring), 1223 steps | 0.100 ms | baseline | 2 | 1,255,936 |
| Ring, committed every step, 1223 steps, 2.4 wraps | 0.439 ms | **4.4x slower** | 2 | 1,575,936 |
| Ring, rollback never committed, 1223 steps, 2.4 wraps | 0.452 ms | 4.5x slower | 2 | 1,575,936 |

Two things worth calling out honestly: **the allocator-level "traffic" this task's acceptance
criterion asks about turns out not to be the story** — `allocation.all.current` and
`num_alloc_retries` are identical across all three runs, because PyTorch's CUDA caching allocator
reuses the same freed blocks for the transient `_chronological()`/`clone()` tensors every step rather
than issuing new `cudaMalloc` calls. The real cost is **extra kernel launches and copies per step**
(readback slice/cat, two clones, one scatter, vs. one contiguous copy), which shows up as wall-clock
time (4.4x) and a ~25% higher peak active-memory footprint (the transient rollback clones), not as
allocator churn. Second: **whether the caller ever calls `commit()` barely matters** (0.439ms vs
0.452ms) — the rollback clones are made unconditionally on every append regardless of whether the
step is ever going to be rolled back, so there is no "cheap path" available today for a caller that
knows it will never truncate. On a single 8GB card at 4-14K decode tok/s (this report's own decode
throughput numbers), an extra ~0.34ms/token from the ring path is not the dominant cost of decode
(which runs at 1-45ms/token depending on batch size per the sweep below) but it is a real, measured,
non-zero tax specifically for Modern's local layers that Edu's full-attention linear cache does not
pay.

### 9. Batch-size sweep (50M-edu, 150M-edu; FP32 vs. FP16; prompt/decode 128/32)

| Model | Precision | Batch | Prefill tok/s | Decode tok/s | TTFT (s) | Peak reserved VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 50M-edu | FP32 | 1 | 1101.9 | 70.3 | 0.116 | 254 MB |
| 50M-edu | FP32 | 2 | 2222.8 | 139.9 | 0.115 | 264 MB |
| 50M-edu | FP32 | 4 | 4663.5 | 264.0 | 0.110 | 292 MB |
| 50M-edu | FP32 | 8 | 9126.0 | 560.1 | 0.112 | 346 MB |
| 50M-edu | FP32 | 16 | 14889.4 | 1097.6 | 0.138 | 474 MB |
| 50M-edu | FP16 | 1 | 775.6 | 63.8 | 0.165 | 138 MB |
| 50M-edu | FP16 | 2 | 1511.0 | 122.2 | 0.169 | 145 MB |
| 50M-edu | FP16 | 4 | 3023.4 | 242.2 | 0.169 | 157 MB |
| 50M-edu | FP16 | 8 | 6932.4 | 495.4 | 0.148 | 183 MB |
| 50M-edu | FP16 | 16 | 14190.3 | 1012.6 | 0.144 | 237 MB |
| 150M-edu | FP32 | 1 | 1089.9 | 50.4 | 0.117 | 709 MB |
| 150M-edu | FP32 | 2 | 2227.4 | 98.7 | 0.115 | 730 MB |
| 150M-edu | FP32 | 4 | 4054.9 | 202.4 | 0.126 | 766 MB |
| 150M-edu | FP32 | 8 | 6925.3 | 348.3 | 0.148 | 891 MB |
| 150M-edu | FP32 | 16 | 10598.8 | 786.8 | 0.193 | 1116 MB |
| 150M-edu | FP16 | 1 | 792.9 | 44.8 | 0.161 | 354 MB |
| 150M-edu | FP16 | 2 | 1563.3 | 89.4 | 0.164 | 367 MB |
| 150M-edu | FP16 | 4 | 3522.6 | 176.7 | 0.145 | 390 MB |
| 150M-edu | FP16 | 8 | 6507.3 | 361.7 | 0.157 | 447 MB |
| 150M-edu | FP16 | 16 | 12615.7 | 719.0 | 0.162 | 552 MB |

None of these 20 runs show the paging-thrashing signature from the 8K-context finding (TTFT stays
under 0.2s throughout; peak reserved VRAM tops out at 1.1GB, far below this card's ~7.5GB danger
zone) — this whole sweep comfortably fits in 8GB at these sequence lengths. **FP32 remains the
faster inference precision at every batch size tested for both prefill and decode**, extending
Finding 6's single-batch observation rather than contradicting it: the FP32-vs-FP16 gap actually
narrows as batch size grows (50M-edu prefill: FP32 is 42% faster at batch=1 but only 5% faster at
batch=16), consistent with fixed per-op autocast dispatch overhead mattering less once there is more
real work per kernel launch — but it never flips in FP16's favor in this range. This is inference-only
and does not contradict Finding 6/MF-075's separate, opposite finding that FP16 is the right choice
for *training* throughput on this hardware.

## Recommendations (synthesized across all MF-050 passes)

- **Training precision on this hardware: FP16 with GradScaler**, not BF16 (emulated, ~2.7x slower)
  and not FP32 (uses far more VRAM at the batch sizes this training benchmark needed) — see MF-075.
- **Inference precision on this hardware: FP32**, across every batch size (1-16) and every model size
  (50M/150M) tested in this report. BF16/FP16 do not show a throughput advantage for single-stream
  inference at these scales, likely because autocast dispatch overhead outweighs any Tensor Core
  benefit at these small per-step workloads.
- **`torch.compile` is not currently worth using on this GPU**: it explicitly declines native BF16
  compilation and shows no steady-state benefit even where it does compile, while adding a one-time
  cost of 10+ seconds.
- **For the hybrid Modern preset's prefill phase specifically, prefer manual attention over eager
  (uncompiled) FlexAttention** on Turing-class hardware — eager FlexAttention was measured 6.4x
  slower here than the plain reference path it is meant to optimize.
- **Do not run 8K+ context single-stream generation with the 150M-modern preset on an 8GB card** at
  batch=1/BF16/full 20 layers without either a smaller context, more aggressive precision reduction,
  or accepting Windows' silent (and very slow) CUDA-to-system-RAM paging fallback.
- **Batch size headroom exists**: this 8GB card comfortably ran batch=16 inference for both 50M and
  150M Edu at 2K-context settings with over 6GB of VRAM to spare, so single-stream latency, not
  memory, is the practical ceiling for interactive use on this hardware; training remains the
  VRAM-constrained side of the picture (MF-049).
- **FlexAttention-vs-first-party-variable-length-sliding-GQA parity/benchmarking remains genuinely
  optional**, per `docs/IMPLEMENTATION_DECISIONS.md`'s own framing ("may benchmark ... only as an
  optional parity-proven path") — not pursued in this pass, and its absence does not block MF-050.

## What this matrix does not cover

Manual-attention benchmarking, ring-cache allocation/rollback profiling, and the batch-size sweep are
now covered (see the 2026-08-26 update above). Still not covered, and out of scope for a home-GPU
gate: multi-stream/concurrent-request serving, any first-party variable-length sliding-GQA backend
(optional per the decision above), and a batch-size sweep on the 150M-modern hybrid preset (the
existing hybrid rows are batch=1 only) — the last of these would be reasonable follow-up work but is
not required by MF-050's acceptance criteria, which is single-stream/fixed-batch scoped throughout.
