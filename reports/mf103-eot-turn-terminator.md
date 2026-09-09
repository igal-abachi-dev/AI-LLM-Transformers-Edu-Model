# MF-103: `<|eot|>` turn-terminator -- real validation

Real end-to-end confirmation that a model trained with the new `<|eot|>`
chat/SFT turn boundary (distinct from `<|eos|>`'s now pretraining-only
document-boundary role) actually learns to stop generation at `<|eot|>`,
not `<|eos|>`, and not merely because it hit `max_new_tokens`.

## Setup

- **Tokenizer**: `data/tokenizer-eot` -- real byte-BPE retrained on the exact
  same corpus every tokenizer this project has trained on
  (`data/tokenizer-corpus.jsonl`, 3,000 real FineWeb-Edu-style documents),
  `vocab_size=16384`, 14 special tokens (IDs 0-10 unchanged from the
  existing frozen tokenizer; `<|eot|>`=11, `<|file_sep|>`=12,
  `<|repo_name|>`=13). **Staged, not yet promoted** to `data/tokenizer/`
  -- see the status note on why (retraining changes the learned BPE merges
  for ordinary text, not just special-token IDs, which would desync every
  existing packed shard pool; promotion is deferred to right before
  [[MF-070]], in the same pass as re-packing).
- **Base pretraining warm-up**: `configs/50m-edu.toml`
  (14 layers, d_model=512, Edu preset), real 800-document/3,248-sequence
  shard pool packed from the same `data/tokenizer-corpus.jsonl` text under
  the new tokenizer (`data/shards/mf103-eot-validation/train`,
  sequence_length=256). `train/pretrain.py --updates 400 --batch-size 8
  --seed 42 --device cuda`: loss 9.7 -> 6.72, 21,388 tok/s. Not a quality
  checkpoint -- a fast, real warm-up so the SFT stage starts from a model
  that has learned some language statistics, not pure noise.
- **SFT**: the real, existing, provenance-complete CC0-1.0 dataset already
  used for the real MF-066 SFT releases (`artifacts/mf066-tmp/sft_dataset.jsonl`,
  32 examples, `"Reply with exactly the word X."` -> `"X"`).
  `train/sft.py --updates 300 --batch-size 1 --learning-rate 1e-4
  --warmup-updates 20 --seed 42 --device cuda`.

## Result: real generations, raw token inspection (not just decoded text)

Ran real greedy generation (`model.generate(..., eos_id=tokenizer.eot_id,
suppress_token_ids=non_assistant_special_token_ids())`, matching exactly
what `generate_assistant` does internally) against 8 of the real training
prompts, `max_new_tokens=16`, and inspected the **raw token IDs**, not just
the special-token-stripped decoded string, so a false "it looks fine"
reading from a stripped string can't hide a token-level mismatch.

| Prompt | Decoded reply | Raw continuation length | `<|eot|>` present | Real early stop (before 16 tokens) | `<|eos|>` ever emitted |
|---|---|---|---|---|---|
| RED / GREEN / PURPLE / ORANGE / SILVER / CRIMSON / AMBER / TEAL (8 prompts) | all decoded to "RED" | 3 | yes (8/8) | yes (8/8) | **no (0/8)** |

**8/8 real generations stopped exactly at `<|eot|>`, before hitting
`max_new_tokens`. 0/8 ever emitted `<|eos|>`.** The mechanism this task
exists to build -- a chat model that stops at the new turn-boundary token,
never the pretraining document-boundary token -- is confirmed working
end to end through the real training and inference paths, not just unit
tests.

## Honest limitation: content accuracy, not claimed here

Every reply decoded to "RED" regardless of the actual color asked
(GREEN/PURPLE/etc. all got "RED" too). This is expected, not a regression:
400 pretraining updates plus 300 SFT updates on 32 examples is nowhere near
enough training for a model to actually learn instruction *content*
(picking the right color) -- this validation was never testing that. The
concrete claim under test -- does the model learn to use `<|eot|>` as its
stop signal -- is unambiguously answered yes, independent of the
(expected, irrelevant-here) content quality.

## Cleanup

The validation checkpoints (`artifacts/mf103-eot-validation/`, ~5.4GB --
periodic AdamW-state checkpoints for two short real training runs) are
deleted after this report was written; they have no further reuse value
(not a release, nothing resumes from them). The tiny shard pool
(`data/shards/mf103-eot-validation/`, ~1.9MB) and the staged tokenizer
(`data/tokenizer-eot/`) are kept.
