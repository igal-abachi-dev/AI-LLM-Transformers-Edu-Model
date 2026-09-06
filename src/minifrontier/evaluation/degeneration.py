"""Repetition/degeneration metrics for generated text (MF-091).

Beginner's map of this file
----------------------------
Cross-entropy and perplexity measure whether the model assigns high probability
to the *right* next token when it is told the true continuation. Neither one
tells you what actually happens once the model has to keep picking its own next
token, over and over, with no correction -- which is exactly the situation real
generation puts it in. A small model under greedy or lightly-tempered decoding
can fall into a loop (repeating the same phrase, or the same few tokens) while
still scoring a perfectly reasonable loss on held-out text, because the loss
never had to grade a self-generated continuation.

``distinct_n`` is the standard, simple way to catch this: chop each generated
sequence into overlapping n-grams and ask what fraction of them are actually
distinct. A healthy generation has most of its 4-grams appear exactly once
(``distinct_4`` close to 1.0); a degenerate loop repeats the same handful of
4-grams over and over (``distinct_4`` close to 0.0). This is the number that
tracks whether the released model's chat demo looks broken, in a way no
lm-eval accuracy task does -- see this project's own min-p/repetition-penalty
additions in ``generation.py``, which this metric exists to help tune.
"""

from __future__ import annotations

from collections.abc import Sequence


def distinct_n(token_sequences: Sequence[Sequence[int]], n: int) -> float:
    """Fraction of unique n-grams among all n-grams across the given sequences.

    ``1.0`` means every n-gram that occurred, occurred exactly once (no
    detectable repetition at this n). Values close to ``0.0`` mean the same
    handful of n-grams are being repeated constantly -- the generation is
    degenerate. Pooling n-grams across every sequence (rather than averaging
    a per-sequence score) matches the metric's standard definition and avoids
    letting one very short, accidentally-diverse sequence skew the average.
    """

    if n < 1:
        raise ValueError("n must be at least 1")
    total = 0
    unique: set[tuple[int, ...]] = set()
    for sequence in token_sequences:
        tokens = list(sequence)
        for start in range(len(tokens) - n + 1):
            total += 1
            unique.add(tuple(tokens[start : start + n]))
    if total == 0:
        raise ValueError(
            "no n-grams available -- every sequence is shorter than n, or none were given"
        )
    return len(unique) / total


def repeated_ngram_fraction(token_sequences: Sequence[Sequence[int]], n: int) -> float:
    """The complement of ``distinct_n``: the fraction of n-grams that are
    repeats of an n-gram seen earlier. Provided alongside ``distinct_n``
    because some reporting conventions prefer "how much is repeated" over
    "how much is distinct" -- they carry the same information."""

    return 1.0 - distinct_n(token_sequences, n)
