"""Synthetic long-context needle-in-haystack retrieval eval (MF-086).

Beginner's map of this file
----------------------------
Cross-entropy and perplexity measure whether a model predicts the *next*
token well on ordinary text. They say nothing about whether a model can
actually *use* information from far back in its context window -- a model
can score a good loss while still failing to retrieve a fact stated 1,500
tokens earlier if attention to distant positions is effectively broken.

The standard way to test this directly (Kamradt's "needle in a haystack"
test, used across the field) is: bury one distinctive, unguessable fact (the
"needle") somewhere inside a long stretch of unrelated filler text (the
"haystack"), then ask the model to state that fact at the very end of the
prompt. If the model's own greedy continuation contains the exact needle
value, retrieval at that context length and needle position succeeded; if
not, it failed. Running this at several needle positions (early/middle/late)
and several total context lengths turns a single yes/no test into a real
map of where a model's long-context retrieval actually breaks down, if it
does at all.

This is a real, if simplified, empirical retrieval test, not a proxy for
open-ended long-context "understanding" -- it only tells you whether the
mechanism can find a stated fact again, which is a real and necessary
(though not sufficient) part of using long context at all.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

from minifrontier.generation import generate
from minifrontier.model import MiniFrontier
from minifrontier.tokenizer import MiniFrontierTokenizer

FILLER_SENTENCE = (
    "The grass is green and the sky is blue on a calm summer afternoon in the countryside. "
)
NEEDLE_TEMPLATE = "The secret code for today is {code}. Remember this code carefully. "
QUESTION = "\n\nWhat is the secret code for today? The secret code for today is"


@dataclass(frozen=True, slots=True)
class NeedleTrial:
    context_length: int
    needle_fraction: float
    code: str
    found: bool
    completion: str


def _random_code(rng: random.Random) -> str:
    """A 6-digit code, unguessable and never appearing in the filler text."""

    return "".join(str(rng.randint(0, 9)) for _ in range(6))


def build_needle_prompt(
    tokenizer: MiniFrontierTokenizer,
    *,
    context_length: int,
    needle_fraction: float,
    code: str,
) -> list[int]:
    """Build a real prompt of (approximately) `context_length` tokens with the
    needle sentence inserted at `needle_fraction` of the way through the filler,
    followed by the retrieval question. Returns real token IDs, not text, so
    the caller controls length exactly in the units that matter (tokens)."""

    if not 0.0 <= needle_fraction <= 1.0:
        raise ValueError("needle_fraction must be in [0, 1]")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    question_ids = tokenizer.encode(QUESTION)
    needle_ids = tokenizer.encode(NEEDLE_TEMPLATE.format(code=code))
    filler_ids = tokenizer.encode(FILLER_SENTENCE)
    budget = context_length - len(question_ids) - len(needle_ids)
    if budget < 0:
        raise ValueError(
            f"context_length {context_length} is too short to hold the needle and question"
        )
    needle_position = int(budget * needle_fraction)
    before = []
    while len(before) < needle_position:
        before.extend(filler_ids)
    before = before[:needle_position]
    after = []
    while len(after) < budget - needle_position:
        after.extend(filler_ids)
    after = after[: budget - needle_position]
    return [*before, *needle_ids, *after, *question_ids]


def run_needle_trial(
    model: MiniFrontier,
    tokenizer: MiniFrontierTokenizer,
    *,
    context_length: int,
    needle_fraction: float,
    seed: int,
    max_new_tokens: int = 12,
) -> NeedleTrial:
    """One real trial: build a prompt, generate greedily, check the code appears."""

    rng = random.Random(seed)
    code = _random_code(rng)
    prompt_ids = build_needle_prompt(
        tokenizer, context_length=context_length, needle_fraction=needle_fraction, code=code
    )
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = generate(model, prompt, max_new_tokens=max_new_tokens, temperature=0)
    completion = tokenizer.decode(
        generated[0, prompt.shape[1] :].tolist(), skip_special_tokens=True
    )
    found = re.search(re.escape(code), completion) is not None
    return NeedleTrial(
        context_length=context_length,
        needle_fraction=needle_fraction,
        code=code,
        found=found,
        completion=completion,
    )


def run_needle_haystack_eval(
    model: MiniFrontier,
    tokenizer: MiniFrontierTokenizer,
    *,
    context_lengths: list[int],
    needle_fractions: list[float],
    seed: int = 0,
    max_new_tokens: int = 12,
) -> list[NeedleTrial]:
    """The full grid: one real trial per (context_length, needle_fraction) pair."""

    if not context_lengths or not needle_fractions:
        raise ValueError("context_lengths and needle_fractions must both be non-empty")
    trials = []
    for length in context_lengths:
        for fraction in needle_fractions:
            trial_seed = hash((seed, length, fraction)) & 0xFFFFFFFF
            trials.append(
                run_needle_trial(
                    model,
                    tokenizer,
                    context_length=length,
                    needle_fraction=fraction,
                    seed=trial_seed,
                    max_new_tokens=max_new_tokens,
                )
            )
    return trials
