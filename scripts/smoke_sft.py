"""Persist a bounded CPU assistant-only SFT overfit and chat-path record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minifrontier.chat import ChatMessage, render_chat
from minifrontier.config import ModelConfig
from minifrontier.evaluation.sft import score_sft_responses
from minifrontier.model import MiniFrontier
from minifrontier.sft import (
    ConversationRecord,
    conversation_hash,
    encode_sft_example,
    pack_sft_examples,
)
from minifrontier.tokenizer import train_byte_bpe
from minifrontier.training import ListBatchProvider, TrainingConfig, train_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/m8-sft-cpu-smoke.json"))
    parser.add_argument("--updates", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(58)
    messages = (ChatMessage("user", "2+2?"), ChatMessage("assistant", "4"))
    tokenizer = train_byte_bpe(
        ["2+2? 4", "user assistant arithmetic answer"],
        vocab_size=320,
        min_frequency=1,
    )
    record = ConversationRecord(
        messages,
        "original-smoke",
        "v1",
        "CC0-1.0",
        "arithmetic-1",
        conversation_hash(messages),
    )
    example = encode_sft_example(record, tokenizer, max_length=32)
    batch = next(pack_sft_examples([example], sequence_length=32, pad_id=tokenizer.pad_id))
    config = ModelConfig.tiny_edu(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_ff=32,
    )
    model = MiniFrontier(config)
    training = TrainingConfig(
        max_updates=args.updates,
        learning_rate=2e-2,
        min_learning_rate=2e-3,
        warmup_updates=min(5, args.updates - 1),
        weight_decay=0.0,
        gradient_clip=10.0,
    )
    _, _, state, _ = train_updates(model, ListBatchProvider([batch]), training)
    report = {
        "status": "bounded_engineering_smoke",
        "quality_claim": False,
        "config": config.to_dict(),
        "updates": state.completed_updates,
        "target_tokens": state.consumed_target_tokens,
        "final_loss": state.last_loss,
        "assistant_target_tokens_per_pack": int(batch.loss_mask.sum()),
        "rendered_template": render_chat(messages),
        "reference_scorer_check": score_sft_responses(
            [
                {"id": "exact", "category": "instruction", "required_substrings": ["4"]},
                {"id": "missing", "category": "unknown", "required_substrings": []},
            ],
            {"exact": "4"},
        ),
        "limitations": [
            "One original example proves masking/learning integration only.",
            "It is not evidence of general instruction-following quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
