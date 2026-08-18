"""Deterministic base completion and bounded multi-turn chat helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from jinja2 import Environment, StrictUndefined

from minifrontier.model import MiniFrontier
from minifrontier.tokenizer import SPECIAL_TOKEN_IDS, MiniFrontierTokenizer

Role = Literal["system", "user", "assistant"]
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).parents[2] / "templates" / "system_prompt.md"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported chat role: {self.role}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("chat message content must be non-empty")


def load_system_prompt(path: str | Path | None = None) -> str:
    """Load a non-empty system prompt without making it part of model correctness."""

    prompt_path = Path(path) if path is not None else DEFAULT_SYSTEM_PROMPT_PATH
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"system prompt is empty: {prompt_path}")
    return prompt


def validate_messages(
    messages: list[ChatMessage] | tuple[ChatMessage, ...],
    *,
    generation_prompt: bool = False,
) -> None:
    if not messages:
        raise ValueError("conversation must contain at least one message")
    offset = 1 if messages[0].role == "system" else 0
    if any(message.role == "system" for message in messages[1:]):
        raise ValueError("system message is allowed only at the beginning")
    turns = messages[offset:]
    if not turns or turns[0].role != "user":
        raise ValueError("conversation must begin with a user turn after optional system")
    for index, message in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.role != expected:
            raise ValueError(f"expected {expected} at turn {index}, got {message.role}")
    if generation_prompt and turns[-1].role != "user":
        raise ValueError("generation prompt requires the conversation to end with a user turn")


def render_chat(
    messages: list[ChatMessage] | tuple[ChatMessage, ...],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render the checked-in Jinja template exactly."""

    validate_messages(messages, generation_prompt=add_generation_prompt)
    template_path = Path(__file__).parents[2] / "templates" / "chat_template.jinja"
    environment = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = environment.from_string(template_path.read_text(encoding="utf-8"))
    return template.render(
        messages=[asdict(message) for message in messages],
        add_generation_prompt=add_generation_prompt,
    )


def encode_chat_prompt(
    tokenizer: MiniFrontierTokenizer,
    messages: list[ChatMessage] | tuple[ChatMessage, ...],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    validate_messages(messages, generation_prompt=add_generation_prompt)
    token_ids = [tokenizer.bos_id]
    for message in messages:
        token_ids.append(SPECIAL_TOKEN_IDS[f"<|{message.role}|>"])
        token_ids.extend(tokenizer.encode("\n" + message.content))
        token_ids.append(tokenizer.eos_id)
        token_ids.extend(tokenizer.encode("\n"))
    if add_generation_prompt:
        token_ids.append(SPECIAL_TOKEN_IDS["<|assistant|>"])
        token_ids.extend(tokenizer.encode("\n"))
    return token_ids


def fit_messages_to_context(
    tokenizer: MiniFrontierTokenizer,
    messages: list[ChatMessage],
    *,
    max_prompt_tokens: int,
) -> tuple[list[ChatMessage], list[int]]:
    """Drop oldest complete user/assistant pairs; never slice a serialized message."""

    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    kept = list(messages)
    validate_messages(kept, generation_prompt=True)
    while True:
        token_ids = encode_chat_prompt(tokenizer, kept, add_generation_prompt=True)
        if len(token_ids) <= max_prompt_tokens:
            return kept, token_ids
        system_offset = 1 if kept[0].role == "system" else 0
        # Preserve the newest user request and remove only a complete older pair.
        if len(kept) - system_offset < 3:
            raise ValueError("latest complete chat turn does not fit the model context")
        del kept[system_offset : system_offset + 2]


def complete_text(
    model: MiniFrontier,
    tokenizer: MiniFrontierTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    seed: int = 42,
) -> str:
    token_ids = tokenizer.encode(prompt, add_bos=True)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=model.token_embedding.weight.device)
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    generated = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
        generator=generator,
    )
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def generate_assistant(
    model: MiniFrontier,
    tokenizer: MiniFrontierTokenizer,
    messages: list[ChatMessage],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    seed: int = 42,
) -> str:
    if max_new_tokens <= 0 or max_new_tokens >= model.config.max_seq_len:
        raise ValueError("max_new_tokens must leave room for a non-empty chat prompt")
    _, prompt_ids = fit_messages_to_context(
        tokenizer,
        messages,
        max_prompt_tokens=model.config.max_seq_len - max_new_tokens,
    )
    prompt = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=model.token_embedding.weight.device,
    )
    generator = torch.Generator(device=prompt.device).manual_seed(seed)
    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
        generator=generator,
    )
    continuation = generated[0, prompt.shape[1] :].tolist()
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()
