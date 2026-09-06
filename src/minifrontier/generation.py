"""KV-cached greedy, temperature, top-k, and nucleus generation.

Beginner's map of this file
---------------------------
This is the loop that turns a language model into something that writes text:

1. Run the prompt through the model once -- the **prefill** -- and keep only the
   final position's scores, which say what should come next.
2. Pick one token from those scores (``sample_next_token``).
3. Append it, feed *only that one new token* back in, and go to step 2.

Step 3 is cheap solely because of the KV cache: every earlier token's Key and
Value are already stored, so a decode step costs one token of work rather than
re-reading the entire conversation. A 500-word answer is roughly 700 trips around
this loop.

Nothing here changes what the model knows. ``temperature``, ``top_k`` and
``top_p`` only change how adventurously we pick from scores the model already
produced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from minifrontier.cache import KVCache

if TYPE_CHECKING:
    from minifrontier.model import MiniFrontier


def _apply_repetition_penalty(
    logits: torch.Tensor, previous_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style: divide positive logits, multiply negative ones, for every
    token already seen anywhere in that row's history. Discourages drift
    without an outright ban, unlike no-repeat-ngram below."""

    seen = torch.zeros_like(logits, dtype=torch.bool)
    seen.scatter_(-1, previous_tokens, True)
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)


def _apply_no_repeat_ngram(
    logits: torch.Tensor, previous_tokens: torch.Tensor, ngram_size: int
) -> torch.Tensor:
    """Hard-ban whichever token would complete an n-gram already seen earlier
    in the same sequence -- the fix for actual loops, as opposed to the softer
    drift `_apply_repetition_penalty` addresses. Per-row Python loop: batches
    here are small (interactive generation), and a variable-length n-gram
    lookup does not vectorize cleanly enough to be worth obscuring this in
    exchange for it."""

    if previous_tokens.shape[1] < ngram_size - 1:
        return logits
    blocked = logits.clone()
    for row in range(logits.shape[0]):
        sequence = previous_tokens[row].tolist()
        prefix = tuple(sequence[-(ngram_size - 1) :]) if ngram_size > 1 else ()
        banned: set[int] = set()
        for start in range(len(sequence) - ngram_size + 1):
            if tuple(sequence[start : start + ngram_size - 1]) == prefix:
                banned.add(sequence[start + ngram_size - 1])
        for token_id in banned:
            blocked[row, token_id] = float("-inf")
    return blocked


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int | None = None,
    previous_tokens: torch.Tensor | None = None,
    suppress_token_ids: Sequence[int] | None = None,
    generator: torch.Generator | None = None,
    validate_logits: bool = False,
) -> torch.Tensor:
    """Turn one row of vocabulary scores per sequence into one chosen token ID.

    * ``temperature = 0`` -- always take the highest-scoring token. Deterministic,
      repetitive, and the default here.
    * ``temperature < 1`` -- sharpen the odds: safer and more predictable.
    * ``temperature > 1`` -- flatten them: more variety, more mistakes.
    * ``top_k`` -- never consider more than the k best candidates.
    * ``top_p`` -- consider the best candidates whose probabilities sum to p
      ("nucleus" sampling). A small set when the model is confident, a large one
      when it is not, which is why it usually beats a fixed ``top_k``.
    * ``min_p`` (Nguyen et al., arXiv:2407.01082) -- drop any candidate whose
      probability is below ``min_p`` times the top candidate's probability.
      Unlike a fixed ``top_p``, the threshold scales with how confident the
      model is at this step -- important at this project's scale, where a
      small model is uncertain far more often than the 1B-123B models the
      paper evaluated, so a fixed top-p is either too permissive or too strict
      at almost every step.
    * ``repetition_penalty``/``no_repeat_ngram_size`` -- need ``previous_tokens``
      (everything generated so far in that row, ``[batch, seq_so_far]``) to have
      any effect; see the two module-level helpers above for what each does.
    * ``suppress_token_ids`` -- forced to ``-inf`` before anything else, so a
      chat generation can never sample e.g. a reserved role marker mid-response.
      Applied even under greedy decoding, not just sampling.

    ``min_p``, ``top_k``, and ``top_p`` compose, and all three run before the
    final softmax, in that order.
    """

    if logits.ndim != 2:
        raise ValueError("sampling logits must be [batch, vocab]")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if not 0.0 <= min_p <= 1.0:
        raise ValueError("min_p must be in [0, 1]")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if no_repeat_ngram_size is not None and no_repeat_ngram_size < 2:
        raise ValueError("no_repeat_ngram_size must be at least 2 when provided")
    if validate_logits and not torch.isfinite(logits).all():
        raise ValueError("sampling logits contain non-finite values")

    # Never mutate the caller's tensor -- it may be the raw model output.
    working = logits.clone()
    if suppress_token_ids:
        working[:, list(suppress_token_ids)] = float("-inf")
    if previous_tokens is not None:
        if repetition_penalty != 1.0:
            working = _apply_repetition_penalty(working, previous_tokens, repetition_penalty)
        if no_repeat_ngram_size is not None:
            working = _apply_no_repeat_ngram(working, previous_tokens, no_repeat_ngram_size)

    # Greedy decoding, handled separately: dividing by zero is undefined, and
    # argmax needs no probabilities at all. Suppression/repetition control
    # above still applies -- greedy plus a repetition penalty is a real,
    # common combination, not just a sampling-only feature.
    if temperature == 0:
        return working.argmax(dim=-1, keepdim=True)

    # Dividing the scores stretches or squashes the gaps between them, which
    # softmax then turns into a flatter or sharper distribution.
    filtered = working.float() / temperature
    if min_p > 0.0:
        probabilities_for_floor = torch.softmax(filtered, dim=-1)
        floor = min_p * probabilities_for_floor.amax(dim=-1, keepdim=True)
        filtered = filtered.masked_fill(probabilities_for_floor < floor, float("-inf"))
    if top_k is not None and top_k < filtered.shape[-1]:
        # The k-th best score; anything below it is set to -inf, i.e. impossible.
        threshold = torch.topk(filtered, top_k, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1.0:
        # Nucleus filtering: sort best-first and keep candidates while the running
        # total is still below p.
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        # Subtracting each token's own probability makes the test exclusive, so the
        # candidate that crosses the threshold is KEPT. That also guarantees the
        # single most likely token always survives, even if it alone exceeds p.
        remove = cumulative - sorted_probabilities >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        # Scatter back into vocabulary order, so the index we sample is a real
        # token ID rather than a position in the sorted list.
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    # Scores -> percentages -> one weighted random draw per sequence.
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: MiniFrontier,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int | None = None,
    suppress_token_ids: Sequence[int] | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    validate_logits: bool = False,
) -> torch.Tensor:
    """Continue ``prompt`` for up to ``max_new_tokens`` tokens.

    Returns prompt and continuation together, ``[batch, prompt + generated]``.
    Generation stops early when every sequence in the batch has produced
    ``eos_id``. The model is put in eval mode for the duration and restored
    afterwards, so a caller mid-training does not silently lose its mode.

    ``min_p``/``repetition_penalty``/``no_repeat_ngram_size``/``suppress_token_ids``
    all pass straight through to ``sample_next_token`` -- see there for what
    each does. ``repetition_penalty``/``no_repeat_ngram_size`` are computed
    from everything generated so far (prompt included), not just this run's
    own continuation.
    """

    if prompt.ndim != 2 or prompt.shape[1] == 0:
        raise ValueError("prompt must be non-empty [batch, sequence] tokens")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus requested tokens exceeds model max_seq_len")
    if eos_id is not None and not 0 <= eos_id < model.config.vocab_size:
        raise ValueError("eos_id is outside the model vocabulary")
    if max_new_tokens == 0:
        return prompt.clone()

    was_training = model.training
    model.eval()
    try:
        # Capacity is exactly what this call can possibly need, so nothing is
        # over-allocated. `bounded_local` gives a hybrid model's local layers a
        # fixed-size ring buffer instead of a cache that grows forever -- that,
        # plus GQA, is what makes Modern's cache several times smaller than Edu's.
        cache = KVCache.allocate(
            model.config,
            batch_size=prompt.shape[0],
            device=prompt.device,
            dtype=None,
            capacity=prompt.shape[1] + max_new_tokens,
            bounded_local=model.config.attention_pattern == "hybrid",
        )
        # Preallocate the answer and write into it, rather than concatenating a new
        # tensor every step. Same result, no repeated reallocation.
        output = torch.empty(
            (prompt.shape[0], prompt.shape[1] + max_new_tokens),
            dtype=prompt.dtype,
            device=prompt.device,
        )
        output[:, : prompt.shape[1]].copy_(prompt)
        output_length = prompt.shape[1]
        # PREFILL: the entire prompt in one pass, which fills the cache for every
        # layer. `logits_to_keep=1` skips the scoreboard for all but the final
        # position, whose scores are the only ones we are going to read.
        logits = model(prompt, cache=cache, logits_to_keep=1).logits[:, 0, :]
        # Sequences in a batch finish at different times; this tracks who is done.
        finished = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
        for step in range(max_new_tokens):
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                previous_tokens=output[:, :output_length],
                suppress_token_ids=suppress_token_ids,
                generator=generator,
                validate_logits=validate_logits,
            )
            # Once a sequence has emitted EOS, keep feeding it EOS. The batch must
            # stay rectangular, so finished rows are padded rather than removed.
            if eos_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
                finished |= next_token.squeeze(1).eq(eos_id)
            output[:, output_length : output_length + 1].copy_(next_token)
            output_length += 1
            if step + 1 == max_new_tokens or finished.all():
                break
            # DECODE: one token in, one token's scores out. Everything the model
            # needs about the past is already sitting in the cache, which is why
            # this step costs roughly the same whether we are at token 10 or 500.
            logits = model(next_token, cache=cache, logits_to_keep=1).logits[:, 0, :]
        # Trim the preallocated buffer if EOS ended the run early.
        return output[:, :output_length]
    finally:
        model.train(was_training)
