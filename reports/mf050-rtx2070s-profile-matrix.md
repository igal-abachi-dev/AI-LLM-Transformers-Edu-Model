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

## What this matrix does not cover

Manual-attention-path benchmarking, a dedicated ring-cache allocation/rollback-traffic profiling
pass, and a full batch-size sweep were not run in this pass — this is a first real home-GPU
measurement set, not the complete matrix MF-050's acceptance criteria eventually wants. See
`tasks/backlog.md` MF-050 status note for what remains open.
