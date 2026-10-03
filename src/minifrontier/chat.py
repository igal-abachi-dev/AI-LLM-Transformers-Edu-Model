"""Deterministic base completion and bounded multi-turn chat helpers.

Beginner's map of this file
---------------------------
The chat window is a friendly illusion. The model never sees bubbles, roles, or a
conversation object -- it sees one flat stream of tokens, and the "roles" are
marker tokens inside that stream::

    <|bos|><|system|>You are helpful.<|eot|><|user|>What is 2+2?<|eot|><|assistant|>

Then it is asked its one and only question: what token comes next? Having been
trained on text where ``<|assistant|>`` is followed by assistant-flavoured
writing, it starts producing an answer, and it stops when it emits ``<|eot|>``
(MF-103) -- the chat/SFT turn boundary, distinct from ``<|eos|>``, which keeps
its original, pretraining-only role marking document boundaries.

This module holds the flattening (a Jinja template, so the exact format is data
rather than code), the rules about which turn orders are legal, and the loop that
keeps a multi-turn conversation inside the model's context limit by dropping the
oldest turns. There is no memory beyond that window: what looks like memory in a
chat product is the whole transcript being re-sent every single turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from jinja2 import Environment, StrictUndefined

from minifrontier.cache import KVCache
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads
from minifrontier.speculative_decoding import speculative_generate
from minifrontier.tokenizer import SPECIAL_TOKEN_IDS, MiniFrontierTokenizer

Role = Literal["system", "user", "assistant"]
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).parents[2] / "templates" / "system_prompt.md"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn. Becomes ``<|role|>`` + content + ``<|eot|>`` once serialized."""

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
    """Enforce the turn order the model was trained on.

    An optional system message first, then strict user/assistant alternation.
    This is stricter than it strictly has to be, on purpose. The model only ever
    saw this pattern during SFT, so anything else is untested territory -- and a
    conversation that quietly deviates produces bad output with no error to
    explain it.
    """

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
        token_ids.append(tokenizer.eot_id)
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
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int | None = None,
    seed: int = 42,
    mtp_heads: MTPHeads | None = None,
    stop_strings: Sequence[str] | None = None,
) -> str:
    token_ids = tokenizer.encode(prompt, add_bos=True)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=model.token_embedding.weight.device)
    # Self-speculative decoding (MF-093/MF-105) is only exact for greedy
    # decoding -- its acceptance rule has no probability-ratio correction for
    # sampling yet. Any non-default temperature/top-k/top-p silently falls
    # back to plain decoding rather than producing an approximation under a
    # decoding mode the guarantee doesn't cover. repetition_penalty/
    # no_repeat_ngram_size join that same guard because, unlike min_p (a
    # no-op under greedy -- see sample_next_token's temperature==0 early
    # return), both are applied *before* the greedy argmax and can change
    # which token wins -- min_p needs no guard for the same reason.
    # stop_strings (MF-154) joins the guard too: speculative decoding's
    # exactness guarantee is about matching greedy's *token* output, and
    # never checks decoded text, so combining the two would silently ignore
    # stop_strings rather than honoring it.
    if (
        mtp_heads is not None
        and temperature == 0.0
        and top_k is None
        and top_p == 1.0
        and repetition_penalty == 1.0
        and no_repeat_ngram_size is None
        and stop_strings is None
    ):
        generated, _stats = speculative_generate(
            model, mtp_heads, tokens, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id
        )
        return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    generator = torch.Generator(device=tokens.device).manual_seed(seed)

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    generated = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        eos_id=tokenizer.eos_id,
        generator=generator,
        stop_strings=stop_strings,
        decode=decode if stop_strings else None,
    )
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def non_assistant_special_token_ids() -> list[int]:
    """Every special-token ID except ``<|eot|>``.

    A chat reply should never itself contain role markers, ``<|pad|>``, or
    FIM/tool tokens -- those are structural, not assistant vocabulary.
    ``<|eot|>`` is excluded because it must remain samplable: it's how chat
    generation stops (MF-103). ``<|eos|>`` is suppressed like everything
    else here -- it is purely a pretraining document-boundary marker now, and
    a chat assistant should never emit it.
    """

    return [token_id for token, token_id in SPECIAL_TOKEN_IDS.items() if token != "<|eot|>"]


@dataclass(slots=True)
class ChatCache:
    """Persisted multi-turn KV-cache state (MF-143), a single-user
    "RadixAttention-lite": without this, every `generate_assistant` call
    re-prefills the whole growing conversation from scratch, redoing the same
    transformer work on turns 1..N-1 again on every turn N.

    Carries the real `KVCache` plus exactly which prompt token IDs it was
    built from, since a later call needs both: the cache to extend, and the
    token IDs to check its stored prefix still matches before trusting it --
    edited/truncated history (including `fit_messages_to_context` dropping
    the oldest turn once the window is full) needs a fresh cache, not silent
    corruption of a stale one. See `new_chat_cache`/`generate_assistant`.
    """

    cache: KVCache
    token_ids: list[int]


def new_chat_cache(model: MiniFrontier) -> ChatCache:
    """Allocate a fresh, empty multi-turn cache at the model's full
    `max_seq_len` capacity -- allocated once, reused in place for the whole
    conversation, so it never needs to grow across turns. Matches this
    project's own real, fixed-shape, single-user scope (`MF-067`)."""

    device = model.token_embedding.weight.device
    cache = KVCache.allocate(
        model.config,
        batch_size=1,
        device=device,
        dtype=None,
        capacity=model.config.max_seq_len,
        bounded_local=model.config.attention_pattern == "hybrid",
    )
    return ChatCache(cache, [])


def generate_assistant(
    model: MiniFrontier,
    tokenizer: MiniFrontierTokenizer,
    messages: list[ChatMessage],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int | None = None,
    suppress_token_ids: Sequence[int] | None = None,
    seed: int = 42,
    stop_strings: Sequence[str] | None = None,
    chat_cache: ChatCache | None = None,
) -> str:
    """``chat_cache`` (MF-143, optional): pass the same `ChatCache` (from
    `new_chat_cache`) across turns to avoid re-prefilling the whole
    conversation every time -- only tokens new since the last call (this
    turn's user message, plus any trimming `fit_messages_to_context` already
    did) are actually run through the model; everything already in the cache
    is reused. If the stored prefix no longer matches (history was edited,
    or the context window dropped an old turn), this falls back to a fresh
    prefill using the *same* cache object (reset in place) rather than
    silently reusing stale, mismatched state. Omit for the original
    behavior: every call re-prefills from scratch.
    """

    if max_new_tokens <= 0 or max_new_tokens >= model.config.max_seq_len:
        raise ValueError("max_new_tokens must leave room for a non-empty chat prompt")
    _, prompt_ids = fit_messages_to_context(
        tokenizer,
        messages,
        max_prompt_tokens=model.config.max_seq_len - max_new_tokens,
    )
    if chat_cache is None:
        new_ids = prompt_ids
        cache = None
    else:
        # Longest common prefix, not all-or-nothing: the stored reply text a
        # caller re-appends to `messages` is this function's own *stripped*
        # return value, so re-encoding it does not always reproduce the exact
        # tokens actually sitting in the cache (trimmed leading/trailing
        # whitespace is a real, common case, not just an edge case) -- find
        # exactly where the two sequences actually diverge and reuse
        # everything up to there, rather than discarding a real prefix match
        # just because the very end of it does not round-trip byte-for-byte.
        shared = 0
        for old, new in zip(chat_cache.token_ids, prompt_ids, strict=False):
            if old != new:
                break
            shared += 1
        if shared < chat_cache.cache.length:
            try:
                chat_cache.cache.truncate(shared)
            except ValueError:
                # A hybrid model's local ring-cache layers can only undo
                # still-uncommitted history (commit() runs right after every
                # successful forward call, so this triggers once the ring has
                # wrapped) -- the real, documented LayerKVCache.truncate
                # limit. Fall back to a full reset rather than reusing
                # desynchronized cache contents.
                chat_cache.cache.reset()
                shared = 0
        new_ids = prompt_ids[shared:]
        chat_cache.token_ids = chat_cache.token_ids[:shared]
        cache = chat_cache.cache
    if not new_ids:
        raise ValueError(
            "chat_cache already covers every prompt token -- nothing new to generate from"
        )
    prompt = torch.tensor(
        [new_ids],
        dtype=torch.long,
        device=model.token_embedding.weight.device,
    )
    generator = torch.Generator(device=prompt.device).manual_seed(seed)

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        suppress_token_ids=suppress_token_ids,
        stop_strings=stop_strings,
        decode=decode if stop_strings else None,
        eos_id=tokenizer.eot_id,  # chat turn boundary (MF-103), not the pretraining <|eos|>
        generator=generator,
        cache=cache,
    )
    continuation = generated[0, prompt.shape[1] :].tolist()
    if chat_cache is not None:
        # generate() never feeds the very last sampled token back through the
        # model (nothing downstream needs its logits), so cache.length is
        # always exactly one token behind the full returned sequence -- slice
        # to the cache's own real length rather than assuming the whole
        # continuation was committed, or the next turn's prefix check would
        # trust token_ids that don't actually match what's stored.
        chat_cache.token_ids = (prompt_ids + continuation)[: chat_cache.cache.length]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()
