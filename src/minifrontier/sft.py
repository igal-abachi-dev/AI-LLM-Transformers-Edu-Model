"""Provenance-complete assistant-only supervised fine-tuning examples.

Beginner's map of this file
---------------------------
A model trained only on raw internet text does not answer questions -- it
*continues* them, because that is what internet text does. Turning it into
something that replies takes a second, much smaller training stage: supervised
fine-tuning (SFT), on examples of "user says X, a good assistant replies Y".

The one idea that makes SFT work is the **loss mask**. The whole conversation is
fed through the model, but only the assistant's tokens are graded. Without that,
the model would also be learning to imitate the user -- to produce more questions.

So each example here carries two parallel arrays of the same length::

    token_ids  <|bos|> <|user|> what is 2+2 <|eos|> <|assistant|> 4 <|eos|>
    loss_mask     F       F      F  F  F  F    F         F        T    T

The conversation is flattened into that one stream of tokens by the marker tokens
from ``tokenizer.py`` -- there are no chat "objects" at this level, only text.
"""

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
    """Fingerprint a conversation for deduplication and contamination checks.

    Hashes only role and content, in a canonical JSON form, so the same
    conversation hashes identically no matter how the source file spaced or
    ordered its other fields.
    """

    canonical = json.dumps(
        [{"role": message.role, "content": message.content} for message in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """One training conversation plus its provenance.

    It must end with an assistant turn -- an example whose last word is the user's
    contains nothing for the model to be graded on.
    """

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
    """One encoded conversation: the tokens, and which of them count for learning.

    ``token_ids`` and ``loss_mask`` are always the same length and line up
    position for position. ``True`` means "grade the model here".
    """

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
    """Flatten a conversation into tokens plus the assistant-only loss mask.

    Each turn becomes ``<|role|>`` + content + ``<|eos|>`` + newline. Only an
    assistant turn's content and its closing ``<|eos|>`` are marked True -- the
    ``<|eos|>`` matters, because that is how the model learns when to *stop*.
    Role markers themselves are never graded: predicting whose turn it is next is
    the template's job, not something the model should be guessing.
    """

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
    # `[1:]` because position 0 can never be a target: the next-token shift in
    # loss.py means the first token is only ever an input.
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
    """Pack encoded conversations into fixed-length batches, masks included.

    Same ribbon-and-slice idea as ``data.pack_documents``, with the loss mask cut
    at exactly the same points so the two never drift apart. Unlike pretraining,
    the trailing remainder is kept and padded by default: SFT datasets are small
    enough that throwing away a partial sequence is a real loss.
    """

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
