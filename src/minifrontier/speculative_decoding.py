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

A real limit, found by design review rather than a crash in production: a
*local* (ring) layer's ``truncate`` can only undo its most recent append in
full, not partially -- verified directly (``LayerKVCache.truncate``'s own
docstring and code, and a minimal repro before this module was trusted). Once
a ring layer has ever wrapped past its ``local_window`` capacity, asking it to
keep one of two newly-appended positions and discard the other raises
``ValueError("cannot truncate committed history after ring-cache wrap")``
rather than silently corrupting anything -- which is the right failure mode,
but a genuine gap for a hybrid model's local layers on any generation longer
than ``local_window`` tokens. `speculative_generate` below watches for this
and permanently stops proposing new drafts once any local layer is within one
step of that boundary, falling back to safe plain decoding for the remainder
-- never a crash, just no further speedup past that point. Fixing this for
real (teaching a ring layer to undo only the tail of its last append) is
future work, not attempted here.

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
from minifrontier.generation import sample_next_token
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


def _speculative_append_is_safe(cache: KVCache) -> bool:
    """False once a 2-token speculative append could need an unsupported partial
    undo on any local (ring) layer -- see the module docstring's real-limit note."""

    return not any(layer.ring and cache.length + 2 > layer.capacity for layer in cache.layers)


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
            if not _speculative_append_is_safe(cache):
                # Past the point where a local ring layer could safely undo a
                # partial speculative append (see module docstring) -- fall back
                # to one plain, un-drafted token rather than risk the
                # unsupported-partial-truncate crash. Correctness-preserving,
                # just no speedup for the remainder of this generation.
                plain = model(real_token, cache=cache, logits_to_keep=1)
                assert plain.logits is not None
                real_token = _greedy(plain.logits[:, 0, :])
                if emit(real_token):
                    break
                continue

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


def _sample_and_probabilities(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one temperature-only sample, plus the full distribution it came from.

    The token itself goes through `generation.sample_next_token` unchanged --
    the same tested code path plain (non-speculative) sampled decoding already
    uses (with `top_k=None, top_p=1.0`, matching this function's own scope
    restriction, see `speculative_generate_sampled`'s docstring). The
    probability vector is the one new piece of arithmetic this needs on top:
    `sample_next_token` never returns it, but the rejection-sampling accept
    test below is defined in terms of it.
    """

    token = sample_next_token(
        logits, temperature=temperature, top_k=None, top_p=1.0, generator=generator
    )
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return token, probabilities


def _residual_resample(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """On reject, resample from `normalize(max(0, p - q))`, not from `p` directly.

    This is the real published correction (Leviathan et al. 2023, "Fast
    Inference from Transformers via Speculative Decoding", arXiv:2211.17192,
    Algorithm 1; Chen et al. 2023, "Accelerating Large Language Model Decoding
    with Speculative Sampling", arXiv:2302.01318, Algorithm 2) that makes the
    *overall* process -- draft, accept-or-reject, resample -- exactly
    distributed as `p` (the target model's own real distribution), not merely
    "close to it." Resampling from raw `p` on reject would double-count the
    probability mass already spent accepting draft tokens that agreed with
    `p`, biasing the result toward whatever the draft over-predicted.

    A reject can only happen with `p != q` somewhere, so `residual`'s total
    mass is always positive here in exact arithmetic; `clamp_min` below is
    numerical-safety margin against floating-point cancellation, not a
    correctness patch for a real zero-mass case.

    A citation trap, not a bug, for anyone cross-checking against the papers
    directly: this module's `p`/`q` follow Leviathan et al.'s convention
    (`p` = target, `q` = draft). Chen et al.'s own Algorithm 2 uses the
    letters the other way around (their `q` is the target, `p` the draft) --
    same formula, swapped names. Reading this code next to Chen et al.'s
    equations without noticing the swap makes it look reversed; it isn't.
    """

    residual = (target_probabilities - draft_probabilities).clamp_min(0.0)
    residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(residual, num_samples=1, generator=generator)


def _rejection_sample(
    target_probabilities: torch.Tensor,
    draft_token: torch.Tensor,
    draft_probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, bool]:
    """The one real decision this whole technique adds: accept the draft, or
    reject and resample. Returns ``(token, accepted)``.

    Isolated from `speculative_generate_sampled`'s generation loop specifically
    so its distributional-equivalence guarantee can be tested directly against
    arbitrary `target_probabilities`/`draft_probabilities` pairs
    (`tests/test_speculative_decoding.py`), not only indirectly through a full
    model's forward pass.

    Accept probability is `min(1, p(x)/q(x))` -- clamped, not just compared,
    so a draft token the target likes MORE than the draft did (`p/q > 1`)
    always accepts rather than only "usually."
    """

    p_x = target_probabilities.gather(-1, draft_token)
    q_x = draft_probabilities.gather(-1, draft_token)
    accept_probability = (p_x / q_x.clamp_min(1e-30)).clamp(max=1.0)
    u = torch.rand((1, 1), generator=generator, device=p_x.device, dtype=p_x.dtype)
    if bool((u <= accept_probability).item()):
        return draft_token, True
    return _residual_resample(target_probabilities, draft_probabilities, generator=generator), False


@torch.no_grad()
def speculative_generate_sampled(
    model: MiniFrontier,
    mtp_heads: MTPHeads,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, SpeculativeStats]:
    """Self-speculative decoding for temperature-only sampling (MF-118).

    `speculative_generate` above is exact only for greedy decoding: comparing
    a *sampled* draft token against a differently-sampled verdict would not
    reproduce plain sampled decoding's own randomness. This function instead
    implements exact rejection sampling (same two papers cited in
    `_residual_resample`'s docstring): draw the draft token `x` from its own
    distribution `q`, look up the target's probability `p(x)` for that same
    token, accept with probability `min(1, p(x)/q(x))`, and on reject,
    resample from the corrected residual distribution instead of `p` raw. The
    guarantee this buys is *distributional*, not bit-identical: run many
    times, the emitted token at any position is exactly as if it had been
    drawn from `p` by itself -- proven by the papers above, checked here by a
    real many-sample empirical comparison
    (`tests/test_speculative_decoding.py`), not merely asserted.

    Scoped to **temperature-only** sampling on purpose (no `top_k`/`top_p`):
    truncating the vocabulary before computing `p`/`q` changes the accept-
    ratio math (the residual would need to account for mass the truncation
    itself removed), a real, separate complication left for a later,
    explicitly separate extension rather than bundled into this first cut
    (see MF-118 in `tasks/backlog.md`). Callers wanting top-k/top-p sampling
    must keep using plain (non-speculative) decoding, exactly as
    `chat.py.complete_text` already does for any request this function does
    not cover.

    Everything else -- the KV-cache rollback on reject, the ring-cache-wrap
    safety fallback, the free bonus token after an accept -- reuses
    `speculative_generate`'s own already-tested mechanics unchanged; see that
    function's and the module's docstrings for why each of those is correct.
    """

    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError("prompt must be non-empty [1, sequence] tokens (single stream only)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus requested tokens exceeds model max_seq_len")
    if eos_id is not None and not 0 <= eos_id < model.config.vocab_size:
        raise ValueError("eos_id is outside the model vocabulary")
    if temperature <= 0:
        raise ValueError(
            "speculative_generate_sampled requires temperature > 0 -- use "
            "speculative_generate for greedy (temperature=0) decoding"
        )

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

        # Prefill: the whole prompt in one pass. The first real token is an
        # ordinary, unconditional sample from the target's own distribution --
        # no draft was involved yet, so no accept test applies to it.
        prefill = model(prompt, cache=cache, logits_to_keep=1, return_hidden_states=True)
        assert prefill.logits is not None and prefill.hidden_states is not None
        real_token = sample_next_token(
            prefill.logits[:, 0, :],
            temperature=temperature,
            top_k=None,
            top_p=1.0,
            generator=generator,
        )
        draft_token, draft_probabilities = _sample_and_probabilities(
            mtp_heads.predict(prefill.hidden_states[:, -1, :]),
            temperature=temperature,
            generator=generator,
        )

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
            if not _speculative_append_is_safe(cache):
                # Same real limit as speculative_generate's own docstring --
                # fall back to one plain, un-drafted (but still real,
                # correctly-distributed) sampled token.
                plain = model(real_token, cache=cache, logits_to_keep=1)
                assert plain.logits is not None
                real_token = sample_next_token(
                    plain.logits[:, 0, :],
                    temperature=temperature,
                    top_k=None,
                    top_p=1.0,
                    generator=generator,
                )
                if emit(real_token):
                    break
                continue

            proposed += 1
            chunk = torch.cat([real_token, draft_token], dim=1)
            step = model(chunk, cache=cache, return_hidden_states=True)
            assert step.logits is not None and step.hidden_states is not None
            target_probabilities = torch.softmax(step.logits[:, 0, :].float() / temperature, dim=-1)
            outcome_token, accept = _rejection_sample(
                target_probabilities, draft_token, draft_probabilities, generator=generator
            )
            if accept:
                accepted += 1
                if emit(outcome_token):
                    break
                bonus_token = sample_next_token(
                    step.logits[:, 1, :],
                    temperature=temperature,
                    top_k=None,
                    top_p=1.0,
                    generator=generator,
                )
                if emit(bonus_token):
                    break
                real_token = bonus_token
                draft_token, draft_probabilities = _sample_and_probabilities(
                    mtp_heads.predict(step.hidden_states[:, 1, :]),
                    temperature=temperature,
                    generator=generator,
                )
            else:
                cache.truncate(cache.length - 1)
                real_token = outcome_token
                if emit(real_token):
                    break
                draft_token, draft_probabilities = _sample_and_probabilities(
                    mtp_heads.predict(step.hidden_states[:, 0, :]),
                    temperature=temperature,
                    generator=generator,
                )

        return output[:, :output_length], SpeculativeStats(proposed, accepted)
    finally:
        model.train(was_training)
