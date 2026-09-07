"""Self-speculative decoding using the MTP heads as a free draft model (MF-093).

Beginner's map of this file
----------------------------
Plain autoregressive decoding is memory-bandwidth-bound: producing one token
means reading every weight once, and almost none of that time is spent on
arithmetic (see ``reports/mf050-rtx2070s-profile-matrix.md``). Producing two
tokens in *one* forward pass costs only a little more than producing one,
because the weights only have to be read once either way. Speculative decoding
exploits this: guess a likely next token cheaply, verify the guess and the
"real" token together in a single batched pass, and get two tokens for close
to the price of one whenever the guess was right.

Classic speculative decoding needs a separate, smaller draft model to do the
guessing. This project's MTP heads (``mtp.py``) already predict two steps
ahead as a free byproduct of the *same* forward pass that predicts one step
ahead -- no second model, no extra forward pass, just one more small linear
layer applied to a hidden state that was already computed. That is exactly
what DeepSeek-V3 repurposes its own MTP module for at inference time, and
exactly what this module reproduces, simplified to this project's real
``n_extra_heads=1`` (t+2 only).

The core cycle, given a confirmed real token and a pending draft for the token
after it:

1. Feed ``[real_token, draft_token]`` together in one batched forward pass.
2. Position 0's logits are the model's own, unbiased opinion of what should
   come after ``real_token`` -- call it the *verdict*. Position 0 never
   attended to ``draft_token`` (causal masking), so the verdict cannot be
   contaminated by a wrong guess.
3. If the verdict agrees with ``draft_token``: accept. Both tokens are real,
   the cache is already correctly extended by two positions, and position 1's
   logits/hidden state immediately give a *third* token and a *new* draft with
   no further forward pass.
4. If it disagrees: reject. Roll the cache back by one position
   (``KVCache.truncate``, already built for exactly this "undo the last
   append" case), keep the verdict as the real token, and get a new draft from
   its own hidden state.

For greedy decoding this is an *exact* acceleration, not an approximation: the
verdict is always what the plain (non-speculative) model would have produced
at that position, so the final sequence is identical either way -- only the
number of forward passes needed to produce it differs. This is the property
``tests/test_speculative_decoding.py`` checks first, before anything about
speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from minifrontier.cache import KVCache
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads


@dataclass(frozen=True, slots=True)
class SpeculativeStats:
    """Real counts from one `speculative_generate` call, not estimates."""

    proposed: int
    accepted: int

    @property
    def acceptance_rate(self) -> float:
        if self.proposed == 0:
            return 0.0
        return self.accepted / self.proposed


def _greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy pick, kept separate from `generation.sample_next_token` on purpose.

    Speculative decoding's exactness guarantee (see module docstring) only
    holds for greedy decoding -- accepting a sampled draft against a
    differently-sampled verdict would not reproduce plain sampled decoding's
    own randomness. Temperature/top-k/top-p speculative decoding is a real,
    more complex technique (comparing probability ratios, not raw tokens) and
    is out of scope for this bounded evaluation.
    """

    return logits.argmax(dim=-1, keepdim=True)


@torch.no_grad()
def speculative_generate(
    model: MiniFrontier,
    mtp_heads: MTPHeads,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
) -> tuple[torch.Tensor, SpeculativeStats]:
    """Greedy self-speculative decoding for a single sequence (batch size 1).

    Returns ``(tokens, stats)`` where ``tokens`` is ``[1, prompt_len +
    generated_len]`` -- identical to what greedy `generation.generate` would
    have produced from the same prompt, verified by
    `tests/test_speculative_decoding.py`. Single-stream only: batching
    speculative decoding across rows with independent accept/reject outcomes
    needs a ragged cache-length story this evaluation does not need to solve.
    """

    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError("prompt must be non-empty [1, sequence] tokens (single stream only)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus requested tokens exceeds model max_seq_len")
    if eos_id is not None and not 0 <= eos_id < model.config.vocab_size:
        raise ValueError("eos_id is outside the model vocabulary")

    was_training = model.training
    model.eval()
    try:
        output = torch.empty(
            (1, prompt.shape[1] + max_new_tokens), dtype=prompt.dtype, device=prompt.device
        )
        output[:, : prompt.shape[1]].copy_(prompt)
        output_length = prompt.shape[1]
        if max_new_tokens == 0:
            return output[:, :output_length], SpeculativeStats(proposed=0, accepted=0)

        cache = KVCache.allocate(
            model.config,
            batch_size=1,
            device=prompt.device,
            dtype=None,
            capacity=prompt.shape[1] + max_new_tokens,
            bounded_local=model.config.attention_pattern == "hybrid",
        )
        proposed = 0
        accepted = 0

        # Prefill: the whole prompt in one pass, plus a free first draft from
        # the last position's hidden state.
        prefill = model(prompt, cache=cache, logits_to_keep=1, return_hidden_states=True)
        assert prefill.logits is not None and prefill.hidden_states is not None
        real_token = _greedy(prefill.logits[:, 0, :])
        draft_token = _greedy(mtp_heads.predict(prefill.hidden_states[:, -1, :]))

        def emit(token: torch.Tensor) -> bool:
            """Write one token; return True if generation should stop (EOS or full)."""

            nonlocal output_length
            output[:, output_length : output_length + 1].copy_(token)
            output_length += 1
            done_eos = eos_id is not None and token.item() == eos_id
            return done_eos or output_length == output.shape[1]

        if emit(real_token):
            return output[:, :output_length], SpeculativeStats(proposed, accepted)

        while output_length < output.shape[1]:
            proposed += 1
            chunk = torch.cat([real_token, draft_token], dim=1)
            step = model(chunk, cache=cache, return_hidden_states=True)
            assert step.logits is not None and step.hidden_states is not None
            verdict = _greedy(step.logits[:, 0, :])
            if verdict.item() == draft_token.item():
                accepted += 1
                if emit(draft_token):
                    break
                bonus_token = _greedy(step.logits[:, 1, :])
                if emit(bonus_token):
                    break
                real_token = bonus_token
                draft_token = _greedy(mtp_heads.predict(step.hidden_states[:, 1, :]))
            else:
                cache.truncate(cache.length - 1)
                if emit(verdict):
                    break
                real_token = verdict
                draft_token = _greedy(mtp_heads.predict(step.hidden_states[:, 0, :]))

        return output[:, :output_length], SpeculativeStats(proposed, accepted)
    finally:
        model.train(was_training)
