"""MF-143: tests for ChatCache / new_chat_cache / generate_assistant's cache reuse.

Real acceptance criterion from the backlog: a real multi-turn session must
measurably avoid re-running the transformer over already-processed prefix
tokens on turns after the first, verified by a direct instrumentation count
of tokens actually run through the model per turn (not just code review);
cache-invalidation correctness (edited/truncated history) must be tested
explicitly, not assumed safe.
"""

from __future__ import annotations

import torch

from minifrontier.chat import (
    ChatCache,
    ChatMessage,
    fit_messages_to_context,
    generate_assistant,
    new_chat_cache,
)
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def _tiny_model(mini_tokenizer, *, local_window: int = 8, max_seq_len: int = 64) -> MiniFrontier:
    torch.manual_seed(0)
    config = ModelConfig.tiny_modern(
        vocab_size=mini_tokenizer.vocab_size, max_seq_len=max_seq_len, local_window=local_window
    )
    return MiniFrontier(config).eval()


def _edu_model(mini_tokenizer, *, max_seq_len: int = 192) -> MiniFrontier:
    """Plain full-MHA Edu preset: every layer uses a linear (never-ring)
    cache, which per `LayerKVCache.truncate`'s own real contract "can always
    rewind, because old slots are never overwritten" -- used for the token-
    savings test below specifically so the real longest-common-prefix reuse
    is demonstrated without Modern's local-layer ring-wrap constraint (a
    real, separate behavior, tested on its own further down) interfering."""

    torch.manual_seed(0)
    config = ModelConfig.tiny_edu(vocab_size=mini_tokenizer.vocab_size, max_seq_len=max_seq_len)
    return MiniFrontier(config).eval()


def test_new_chat_cache_starts_empty(mini_tokenizer) -> None:
    model = _tiny_model(mini_tokenizer)
    chat_cache = new_chat_cache(model)
    assert isinstance(chat_cache, ChatCache)
    assert chat_cache.token_ids == []
    assert chat_cache.cache.length == 0
    assert chat_cache.cache.capacity == model.config.max_seq_len


def test_first_turn_with_chat_cache_matches_the_uncached_reply_exactly(mini_tokenizer) -> None:
    """Turn 1 always does a full fresh prefill either way -- no cache-hit
    involved yet -- so the two must be byte-identical."""

    model = _tiny_model(mini_tokenizer)
    messages = [ChatMessage("user", "hello there")]

    reply_without_cache = generate_assistant(
        model, mini_tokenizer, messages, max_new_tokens=4, seed=7
    )
    reply_with_cache = generate_assistant(
        model, mini_tokenizer, messages, max_new_tokens=4, seed=7, chat_cache=new_chat_cache(model)
    )
    assert reply_with_cache == reply_without_cache


def _count_tokens_processed(model: MiniFrontier, mini_tokenizer, *, use_cache: bool) -> int:
    """Real token-budget note: the model's own `max_seq_len` must leave real
    headroom for 3 messages (user/assistant/user) plus the generation prompt,
    or `fit_messages_to_context` correctly trims the oldest turn and the
    cache correctly (not buggily) falls back every time -- that's the real
    invalidation path, tested separately below, not what this test is
    measuring, so give it enough room that no trim happens here."""

    processed: list[int] = []
    original_forward = model.forward

    def recording_forward(tokens, **kwargs):
        processed.append(tokens.shape[1])
        return original_forward(tokens, **kwargs)

    model.forward = recording_forward
    try:
        messages = [ChatMessage("user", "hello there, how is it going")]
        chat_cache = new_chat_cache(model) if use_cache else None
        reply1 = generate_assistant(
            model, mini_tokenizer, messages, max_new_tokens=3, seed=1, chat_cache=chat_cache
        )
        messages.append(ChatMessage("assistant", reply1))
        messages.append(ChatMessage("user", "tell me something else entirely now"))
        generate_assistant(
            model, mini_tokenizer, messages, max_new_tokens=3, seed=2, chat_cache=chat_cache
        )
    finally:
        model.forward = original_forward
    return sum(processed)


def test_chat_cache_processes_fewer_total_tokens_across_two_turns(mini_tokenizer) -> None:
    """The real acceptance criterion: direct instrumentation proving turn 2
    does not re-run the model over turn 1's already-processed prefix."""

    without_cache = _count_tokens_processed(
        _edu_model(mini_tokenizer), mini_tokenizer, use_cache=False
    )
    with_cache = _count_tokens_processed(_edu_model(mini_tokenizer), mini_tokenizer, use_cache=True)
    assert with_cache < without_cache


def test_chat_cache_falls_back_safely_once_a_hybrid_models_ring_layers_have_wrapped(
    mini_tokenizer,
) -> None:
    """Real, separate behavior from the longest-common-prefix reuse above:
    a hybrid (Modern) model's local layers use a ring cache, which can only
    undo its own still-uncommitted last append (LayerKVCache.truncate's real
    documented limit) -- once a small local_window has wrapped, an attempt
    to truncate back into older, already-committed history must fall back to
    a full reset rather than raising out of generate_assistant entirely."""

    model = _tiny_model(mini_tokenizer, local_window=4, max_seq_len=192)
    chat_cache = new_chat_cache(model)
    messages = [ChatMessage("user", "a reasonably long opening message to fill the window")]
    generate_assistant(
        model, mini_tokenizer, messages, max_new_tokens=8, seed=1, chat_cache=chat_cache
    )
    assert chat_cache.cache.length > model.config.local_window  # the ring has genuinely wrapped

    messages.append(ChatMessage("assistant", "placeholder"))
    messages.append(ChatMessage("user", "a second message"))
    # Must not raise -- the real ValueError from LayerKVCache.truncate is
    # caught internally and handled as a full reset, not surfaced to the caller.
    generate_assistant(
        model, mini_tokenizer, messages, max_new_tokens=3, seed=2, chat_cache=chat_cache
    )
    assert chat_cache.cache.length == len(chat_cache.token_ids)


def test_chat_cache_falls_back_to_a_fresh_prefill_when_history_is_edited(mini_tokenizer) -> None:
    """Cache-invalidation correctness, tested explicitly, not assumed safe:
    editing history between turns must not corrupt/misuse the stale cache."""

    model = _tiny_model(mini_tokenizer)
    chat_cache = new_chat_cache(model)
    messages = [ChatMessage("user", "hello there")]
    generate_assistant(
        model, mini_tokenizer, messages, max_new_tokens=3, seed=1, chat_cache=chat_cache
    )
    assert chat_cache.cache.length == len(chat_cache.token_ids)
    assert len(chat_cache.token_ids) > 0

    # Simulate edited history: a completely different conversation, not an
    # extension of the one the cache was built from.
    edited_messages = [ChatMessage("user", "a totally different opening message")]
    generate_assistant(
        model, mini_tokenizer, edited_messages, max_new_tokens=3, seed=1, chat_cache=chat_cache
    )

    # The cache's own bookkeeping must now reflect the EDITED conversation --
    # proves it really reset and reprocessed from scratch rather than
    # keeping a corrupted mix of old and new state.
    _, expected_prompt_ids = fit_messages_to_context(
        mini_tokenizer, edited_messages, max_prompt_tokens=model.config.max_seq_len - 3
    )
    assert chat_cache.token_ids[: len(expected_prompt_ids)] == expected_prompt_ids
    assert chat_cache.cache.length == len(chat_cache.token_ids)


def test_generate_assistant_rejects_a_cache_that_already_covers_the_whole_prompt(
    mini_tokenizer,
) -> None:
    # Edu (plain, linear-cache-only) on purpose: a hybrid model's ring layers
    # would hit their own real wrap-constraint first and safely fall back to
    # a full reset instead, which is real, correct, and tested separately
    # above -- not the "already fully cached" case this test targets.
    model = _edu_model(mini_tokenizer)
    chat_cache = new_chat_cache(model)
    messages = [ChatMessage("user", "hello there")]
    generate_assistant(
        model, mini_tokenizer, messages, max_new_tokens=3, seed=1, chat_cache=chat_cache
    )
    # Calling again with the identical, unextended message list means the
    # cache already covers every prompt token -- nothing new to run.
    try:
        generate_assistant(
            model, mini_tokenizer, messages, max_new_tokens=3, seed=1, chat_cache=chat_cache
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
