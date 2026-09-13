# MF-097: real bounded comparison — ribbon vs. best-fit vs. BOS-aligned-crop packing

Real measurements of `train/pretrain.py` on `150m-modern.toml`, RTX 2070
Super, FP16, real FineWeb-Edu data (documents 5000-47000, the same real
range as the existing ribbon baseline `data/shards/mf064-150m-train`), seed
42, 5000 updates per arm. `best_fit`/`bos_crop` pools built via
`scripts/prepare_data.py --packing {best_fit,bos_crop}` from the identical
document range.

## Real results — corrected, shared-validation-set comparison

**All three checkpoints evaluated against the same shared validation set**
(`data/shards/mf064-150m-train/validation`, ribbon-packed). This matters: a
model's forward pass doesn't care how its *training* data was packed, so
evaluating every checkpoint against one common, fixed set of held-out bytes
is the fair comparison — not each arm's own packing-dependent validation
split (see "A real methodological correction" below for what went wrong
the first time).

| Arm | Cross-entropy | Perplexity | Bits/byte | vs. ribbon (relative CE) | Tokens/second |
|---|---|---|---|---|---|
| ribbon (current default) | 5.100674 | 164.13 | 1.71589 | — | 4,057.9 |
| best_fit | 5.115803 | 166.63 | 1.72098 | +0.297% (6.4x noise floor) | 3,971.8 |
| bos_crop | 5.116401 | 166.73 | 1.72118 | +0.308% (6.7x noise floor) | 4,027.5 |

**Both alternatives are slightly, but really, worse than ribbon — by
almost identical margins.** Real training-data volume from the same
42,000-document range: ribbon 45,789,184 tokens, best_fit 45,790,103 tokens
(essentially identical, as designed), bos_crop 21,306,028 tokens (53.5%
fewer — the real, disclosed cost of cropping, for no offsetting benefit).

## A real methodological correction — the first pass here was wrong

The first version of this comparison evaluated each checkpoint against its
*own* packing-dependent validation split, since packing changes which
packed rows (not which documents — the train/validation split itself is
decided by a hash of document content, independent of packing) end up in
the validation slice. That first pass showed bos_crop with an apparent
**−1.424% CE win** (31.0x noise floor) — a large, seemingly decisive
result. Once re-evaluated on the shared validation set above, that
apparent win **completely disappears and reverses**: bos_crop is actually
+0.308% *worse* than ribbon. The entire −1.424% "win" was an artifact of
bos_crop's own validation split being a smaller (939,263 vs. 1,868,949
bytes), different, and apparently easier sample of held-out text — not a
real training-quality effect. **This is a real, concrete demonstration of
why a shared, packing-independent validation set is required for this
class of comparison, not an optional nicety.**

## Decision

**Ribbon stays the default for [[MF-070]].** Neither alternative earns a
switch: both are real, small, consistent losses (not gains) on the fair
comparison, and bos_crop additionally throws away over half the real
training tokens for that loss. best_fit's throughput is also slightly
worse (3,971.8 vs 4,057.9 tok/s); bos_crop's is roughly on par. This
question is now closed with real, trustworthy evidence — no further
follow-up needed before [[MF-070]].

## Limitations

Single seed, single scale, single token budget, as with every other bounded
comparison this session. The shared-validation-set correction above is
exactly the kind of check this project's own methodology exists to catch
(matching the earlier MF-087 tokenizer/digit-splitting confound) — recorded
here as a real example of it working as intended, not as an embarrassment
to hide.
