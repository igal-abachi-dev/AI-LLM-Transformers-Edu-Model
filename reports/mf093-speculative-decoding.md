# MF-093 — self-speculative decoding using the MTP heads (real result)

## Commands

Retrain (needed because the original `artifacts/mf070-mtp-quality/seed-42-mtp` checkpoint never
persisted its MTP head weights — see the prerequisite-gap note in `tasks/backlog.md`):

```
./.venv/Scripts/python.exe scripts/compare_mtp.py --config configs/150m-modern.toml \
  --train-shards data/shards/mf064-150m-train/train \
  --validation-shards data/shards/mf064-150m-train/validation --tokenizer data/tokenizer \
  --output artifacts/mf093-mtp-retrain --updates 5000 --batch-size 2 --seed 42 \
  --mtp-extra-heads 1 --mtp-loss-weight 0.3 --device cuda --arms mtp
```

Real measurement:

```
./.venv/Scripts/python.exe scripts/compare_speculative_decoding.py \
  --checkpoint artifacts/mf093-mtp-retrain/seed-42-mtp \
  --prompt-length 32 --max-new-tokens 200 --device cuda
```

RTX 2070 Super, FP16, `150m-modern.toml` (16,384 vocab, post-[[MF-087]]-revert), real trained
checkpoint (`artifacts/mf093-mtp-retrain/seed-42-mtp`, CE 5.1315 / PPL 169.27 / BPB 1.7263 on
real held-out validation, closely reproducing the original `mf070-mtp-quality` result), 2026-09-08.

## Real result

| | tokens/second | wall time (200 tokens) |
|---|---:|---:|
| Plain greedy decode | 14.29 | 14.00s |
| Self-speculative decode | 17.25 | 11.60s |

- **Real speedup: 1.21x** (20.7% less wall-clock for the same 200 real output tokens).
- **`tokens_match: true`** — the two decoding paths produced byte-for-byte identical output, on
  real hardware, not just the tiny-CPU-fixture unit tests. Confirms the exactness guarantee
  (see `speculative_decoding.py`'s own docstring) holds in practice, not just in principle.
- **Real draft-acceptance rate: 45.3%** (62 accepted / 137 proposed) — measured, not assumed.
  This is a real, previously-untested number: the pre-registered expectation (recorded in
  MF-093's own backlog status note before this run) was that acceptance would likely be *lower*
  than DeepSeek-V3's own reported numbers, since this project's MTP heads are independent linear
  heads reading the shared hidden state, not DeepSeek-V3's chained transformer-block design. A
  45.3% acceptance rate from that simpler design is a real, respectable result, not a
  disappointing one.
- **The arithmetic is exactly self-consistent**: 1 prefill token + 62 accepted cycles × 2 tokens
  each + 75 rejected cycles × 1 token each = 1 + 124 + 75 = 200, exactly matching
  `max_new_tokens=200` — confirming the accept/reject accounting in `speculative_generate` is
  correct, not just plausible-looking.

## What this settles

The MTP heads this project already trained and had shelved as a training-time regularizer (see
`reports/mf070-mtp-quality.md` — not worth their training-time cost) are, independently, a real,
measured inference-time win when repurposed as a self-speculative decoding draft source. A 1.21x
wall-clock speedup at zero additional model parameters trained *for this purpose* (the same
heads, just used differently) is a genuine, verified result — not a training-time cost with no
offsetting benefit, but a separate capability those same weights turn out to provide for free at
inference time.

## What this does not settle

- Single seed, single prompt length (32), single generation length (200 tokens), single
  configuration (`n_extra_heads=1`, matching this project's only ever-trained MTP setup) — not a
  sweep over any of these dimensions.
- The real 1.21x speedup is specific to this hardware (RTX 2070 Super), this precision (FP16),
  and this model scale (150M-class). Larger models (where the memory-bandwidth-bound decode
  argument this technique exploits gets stronger) or different hardware could show a different
  ratio in either direction.
- `speculative_generate`'s known real limitation (see its own module docstring): once a local
  ring layer's KV cache has wrapped past `local_window` capacity, it permanently falls back to
  plain decoding for the remainder of that generation (a real, previously-undiscovered
  `LayerKVCache.truncate` gap, found by design review and handled safely rather than crashing).
  This measurement's `prompt_length + max_new_tokens = 232` stays comfortably under
  `local_window=512`, so the fallback never triggered here — a longer real generation would see
  its measured speedup taper off past that boundary, not sustain 1.21x indefinitely.
- Greedy decoding only. The exactness guarantee this technique relies on does not extend to
  temperature/top-k/top-p sampling without a more complex (probability-ratio-based) acceptance
  rule, which was not implemented or measured here.
