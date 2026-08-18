"""Provenance-complete assistant-only supervised fine-tuning examples."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from minifrontier.chat import ChatMessage, validate_messages
from minifrontier.tokenizer import SPECIAL_TOKEN_IDS, MiniFrontierTokenizer
from minifrontier.training import TrainingBatch


def conversation_hash(messages: tuple[ChatMessage, ...]) -> str:
    canonical = json.dumps(
        [{"role": message.role, "content": message.content} for message in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    messages: tuple[ChatMessage, ...]
    source: str
    revision: str
    license: str
    record_id: str
    content_hash: str

    def __post_init__(self) -> None:
        validate_messages(self.messages)
        if self.messages[-1].role != "assistant":
            raise ValueError("SFT conversation must end with an assistant turn")
        if not all((self.source, self.revision, self.license, self.record_id)):
            raise ValueError("SFT provenance fields must be non-empty")
        if self.content_hash != conversation_hash(self.messages):
            raise ValueError("conversation content_hash does not match messages")

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> ConversationRecord:
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("conversation messages must be a list")
        messages = tuple(ChatMessage(**message) for message in raw_messages)
        return cls(
            messages=messages,
            source=str(value.get("source", "")),
            revision=str(value.get("revision", "")),
            license=str(value.get("license", "")),
            record_id=str(value.get("record_id", "")),
            content_hash=str(value.get("content_hash", "")),
        )


@dataclass(frozen=True, slots=True)
class SFTExample:
    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    record_id: str


def iter_conversations(path: str | Path) -> Iterator[ConversationRecord]:
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield ConversationRecord.from_mapping(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid conversation at {path}:{line_number}: {error}"
                ) from error


def encode_sft_example(
    record: ConversationRecord,
    tokenizer: MiniFrontierTokenizer,
    *,
    max_length: int,
) -> SFTExample:
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    segments: list[tuple[list[int], list[bool], str]] = []
    for message in record.messages:
        role = SPECIAL_TOKEN_IDS[f"<|{message.role}|>"]
        content = tokenizer.encode("\n" + message.content)
        newline = tokenizer.encode("\n")
        ids = [role, *content, tokenizer.eos_id, *newline]
        assistant = message.role == "assistant"
        mask = [False, *([assistant] * len(content)), assistant, *([False] * len(newline))]
        segments.append((ids, mask, message.role))

    # Remove only complete oldest user/assistant pairs until the example fits.
    while 1 + sum(len(ids) for ids, _, _ in segments) > max_length:
        offset = 1 if segments and segments[0][2] == "system" else 0
        if len(segments) - offset < 4:
            raise ValueError("conversation cannot fit without splitting a message")
        del segments[offset : offset + 2]
    token_ids = [tokenizer.bos_id]
    loss_mask = [False]
    for ids, mask, _ in segments:
        token_ids.extend(ids)
        loss_mask.extend(mask)
    if not any(loss_mask[1:]):
        raise ValueError("SFT example has no assistant prediction targets")
    return SFTExample(tuple(token_ids), tuple(loss_mask), record.record_id)


def pack_sft_examples(
    examples: Iterable[SFTExample],
    *,
    sequence_length: int,
    pad_id: int,
    drop_remainder: bool = False,
) -> Iterator[TrainingBatch]:
    tokens: list[int] = []
    masks: list[bool] = []
    for example in examples:
        tokens.extend(example.token_ids)
        masks.extend(example.loss_mask)
        while len(tokens) >= sequence_length:
            yield TrainingBatch(
                torch.tensor([tokens[:sequence_length]], dtype=torch.long),
                loss_mask=torch.tensor([masks[:sequence_length]], dtype=torch.bool),
            )
            del tokens[:sequence_length]
            del masks[:sequence_length]
    if tokens and not drop_remainder:
        padding = sequence_length - len(tokens)
        yield TrainingBatch(
            torch.tensor([[*tokens, *([pad_id] * padding)]], dtype=torch.long),
            loss_mask=torch.tensor([[*masks, *([False] * padding)]], dtype=torch.bool),
        )
