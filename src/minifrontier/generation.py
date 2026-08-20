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
from typing import TYPE_CHECKING

import torch

from minifrontier.cache import KVCache

if TYPE_CHECKING:
    from minifrontier.model import MiniFrontier


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
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

    ``top_k`` and ``top_p`` compose, and both run before the final softmax.
    """

    if logits.ndim != 2:
        raise ValueError("sampling logits must be [batch, vocab]")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if validate_logits and not torch.isfinite(logits).all():
        raise ValueError("sampling logits contain non-finite values")
    # Greedy decoding, handled separately: dividing by zero is undefined, and
    # argmax needs no probabilities at all.
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    # Dividing the scores stretches or squashes the gaps between them, which
    # softmax then turns into a flatter or sharper distribution.
    filtered = logits.float() / temperature
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
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    validate_logits: bool = False,
) -> torch.Tensor:
    """Continue ``prompt`` for up to ``max_new_tokens`` tokens.

    Returns prompt and continuation together, ``[batch, prompt + generated]``.
    Generation stops early when every sequence in the batch has produced
    ``eos_id``. The model is put in eval mode for the duration and restored
    afterwards, so a caller mid-training does not silently lose its mode.
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
