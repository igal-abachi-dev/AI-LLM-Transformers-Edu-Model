import pytest

from minifrontier.evaluation.degeneration import distinct_n, repeated_ngram_fraction


def test_distinct_n_is_one_for_all_unique_bigrams() -> None:
    # 5 tokens -> 4 bigrams, all distinct.
    assert distinct_n([[1, 2, 3, 4, 5]], n=2) == pytest.approx(1.0)


def test_distinct_n_detects_a_repeating_loop() -> None:
    # "1 2 1 2 1 2" repeats the same two bigrams over and over.
    sequence = [1, 2, 1, 2, 1, 2]
    # bigrams: (1,2),(2,1),(1,2),(2,1),(1,2) -> unique {(1,2),(2,1)} = 2, total 5
    assert distinct_n([sequence], n=2) == pytest.approx(2 / 5)


def test_distinct_n_pools_across_multiple_sequences() -> None:
    # Same bigram (1, 2) repeated across two separate sequences should count
    # as non-distinct when pooled, not be treated as two independent 1.0 scores.
    assert distinct_n([[1, 2], [1, 2]], n=2) == pytest.approx(0.5)


def test_repeated_ngram_fraction_is_the_complement_of_distinct_n() -> None:
    sequence = [1, 2, 1, 2, 1, 2]
    assert repeated_ngram_fraction([sequence], n=2) == pytest.approx(1 - distinct_n([sequence], n=2))


def test_distinct_n_rejects_invalid_n_or_empty_input() -> None:
    with pytest.raises(ValueError, match="n must be at least 1"):
        distinct_n([[1, 2, 3]], n=0)
    with pytest.raises(ValueError, match="no n-grams available"):
        distinct_n([[1]], n=5)
    with pytest.raises(ValueError, match="no n-grams available"):
        distinct_n([], n=2)
