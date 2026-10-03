import pytest
import torch

from minifrontier.config import ModelConfig
from minifrontier.generation import sample_next_token
from minifrontier.model import MiniFrontier


@torch.no_grad()
def naive_greedy(model: MiniFrontier, prompt: torch.Tensor, count: int) -> torch.Tensor:
    output = prompt.clone()
    for _ in range(count):
        token = model(output).logits[:, -1].argmax(dim=-1, keepdim=True)
        output = torch.cat((output, token), dim=1)
    return output


def test_cached_greedy_matches_uncached_reference_and_restores_mode() -> None:
    torch.manual_seed(14)
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=16)).train()
    prompt = torch.tensor([[1, 2, 3]])
    model.eval()
    expected = naive_greedy(model, prompt, 5)
    model.train()
    actual = model.generate(prompt, max_new_tokens=5)
    assert torch.equal(actual, expected)
    assert model.training


def test_generation_rejects_capacity_overflow_instead_of_restarting_positions() -> None:
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=5)).eval()
    with pytest.raises(ValueError, match="exceeds"):
        model.generate(torch.tensor([[1, 2, 3, 4]]), max_new_tokens=2)


def test_sampling_argument_validation_and_seeded_top_k() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert sample_next_token(logits, temperature=0, top_k=None, top_p=1.0).item() == 3
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)
    first = sample_next_token(
        logits.repeat(20, 1),
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        generator=first_generator,
    )
    second = sample_next_token(
        logits.repeat(20, 1),
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        generator=second_generator,
    )
    assert torch.equal(first, second)
    assert set(first.squeeze(1).tolist()) <= {2, 3}
    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(logits, temperature=1, top_k=None, top_p=0)
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(logits, temperature=float("nan"), top_k=None, top_p=1)
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(logits, temperature=float("inf"), top_k=None, top_p=1)
    with pytest.raises(ValueError, match="non-finite"):
        sample_next_token(
            torch.tensor([[float("nan"), 1.0]]),
            temperature=0,
            top_k=None,
            top_p=1,
            validate_logits=True,
        )


def test_suppress_token_ids_applies_even_under_greedy_decoding() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert sample_next_token(logits, temperature=0, top_k=None, top_p=1.0).item() == 3
    suppressed = sample_next_token(
        logits, temperature=0, top_k=None, top_p=1.0, suppress_token_ids=[3]
    )
    assert suppressed.item() == 2
    # Suppressing every candidate but one collapses choice to that one, even
    # under real sampling -- confirms suppression runs before softmax/topk/topp.
    only_zero_allowed = sample_next_token(
        logits.repeat(10, 1),
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        suppress_token_ids=[1, 2, 3],
    )
    assert set(only_zero_allowed.squeeze(1).tolist()) == {0}


def test_min_p_drops_candidates_below_fraction_of_top_probability() -> None:
    # Token 0 dominates the softmax; min_p=0.5 should exclude anything below
    # half of token 0's probability, leaving only token 0 reachable.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    result = sample_next_token(
        logits.repeat(10, 1), temperature=1.0, top_k=None, top_p=1.0, min_p=0.5
    )
    assert set(result.squeeze(1).tolist()) == {0}
    with pytest.raises(ValueError, match="min_p"):
        sample_next_token(logits, temperature=1, top_k=None, top_p=1, min_p=1.5)


def test_repetition_penalty_and_no_repeat_ngram_change_the_greedy_choice() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    # Token 1 is the plain greedy choice.
    assert sample_next_token(logits, temperature=0, top_k=None, top_p=1.0).item() == 1
    # Having already produced token 1 (a positive logit) means it gets divided
    # down by the penalty, past token 2's score.
    previous = torch.tensor([[1, 1]])
    penalized = sample_next_token(
        logits,
        temperature=0,
        top_k=None,
        top_p=1.0,
        repetition_penalty=3.0,
        previous_tokens=previous,
    )
    assert penalized.item() == 2
    with pytest.raises(ValueError, match="repetition_penalty"):
        sample_next_token(logits, temperature=0, top_k=None, top_p=1.0, repetition_penalty=0.0)


def test_no_repeat_ngram_blocks_a_token_that_would_repeat_a_seen_bigram() -> None:
    # History: 5, 1, 2, 1 -- the bigram (1, 2) already occurred once. If the
    # next token would be preceded by another 1, completing (1, 2) again must
    # be blocked, forcing the second-best token instead.
    previous = torch.tensor([[5, 1, 2, 1]])
    logits = torch.tensor([[1.0, 1.0, 9.0, 2.0]])  # token 2 is the greedy choice
    blocked = sample_next_token(
        logits,
        temperature=0,
        top_k=None,
        top_p=1.0,
        no_repeat_ngram_size=2,
        previous_tokens=previous,
    )
    assert blocked.item() != 2
    assert blocked.item() == 3  # next-best surviving candidate
    with pytest.raises(ValueError, match="no_repeat_ngram_size"):
        sample_next_token(logits, temperature=0, top_k=None, top_p=1.0, no_repeat_ngram_size=1)


def test_generate_never_emits_a_suppressed_token(monkeypatch) -> None:
    """End-to-end: even when the model's own logits strongly favor a reserved
    token, generate() with suppress_token_ids must never emit it."""

    model = MiniFrontier(ModelConfig.tiny_edu(vocab_size=16, max_seq_len=8)).eval()
    original_forward = model.forward

    def biased_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        output.logits[..., 5] = 1e9  # token 5 would always win greedily
        return output

    monkeypatch.setattr(model, "forward", biased_forward)
    result = model.generate(torch.tensor([[1, 2]]), max_new_tokens=4, suppress_token_ids=[5])
    assert 5 not in result[:, 2:].tolist()[0]


def test_generation_requests_only_last_logit(monkeypatch) -> None:
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=8)).eval()
    original = model.forward
    requested: list[int | None] = []

    def recording_forward(*args, **kwargs):
        requested.append(kwargs.get("logits_to_keep"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)
    model.generate(torch.tensor([[1, 2]]), max_new_tokens=3)
    assert requested == [1, 1, 1]


def test_per_row_eos_tokens_do_not_change_finished_rows(monkeypatch) -> None:
    model = MiniFrontier(ModelConfig.tiny_edu(vocab_size=16, max_seq_len=8)).eval()
    calls = 0

    def scripted_sample(logits, **kwargs):
        nonlocal calls
        result = torch.tensor([[2], [4]]) if calls == 0 else torch.tensor([[7], [2]])
        calls += 1
        return result

    monkeypatch.setattr("minifrontier.generation.sample_next_token", scripted_sample)
    result = model.generate(torch.tensor([[1], [1]]), max_new_tokens=3, eos_id=2)
    assert result[:, 1:].tolist() == [[2, 2], [4, 2]]


def test_stop_strings_requires_decode() -> None:
    model = MiniFrontier(ModelConfig.tiny_edu(max_seq_len=8)).eval()
    with pytest.raises(ValueError, match="decode"):
        model.generate(torch.tensor([[1, 2]]), max_new_tokens=3, stop_strings=["STOP"])


def test_stop_strings_ends_generation_as_soon_as_the_string_forms_on_a_token_boundary(
    monkeypatch,
) -> None:
    """Aligned case: one token == one character of the stop string."""

    model = MiniFrontier(ModelConfig.tiny_edu(vocab_size=16, max_seq_len=16)).eval()
    # 10='S', 11='T', 12='O', 13='P', 14='x' (would continue past the stop point
    # if stopping failed).
    scripted = [10, 11, 12, 13, 14, 14]
    calls = 0

    def scripted_sample(logits, **kwargs):
        nonlocal calls
        token = torch.tensor([[scripted[calls]]])
        calls += 1
        return token

    char_for = {10: "S", 11: "T", 12: "O", 13: "P", 14: "x"}

    def decode(ids):
        return "".join(char_for[i] for i in ids)

    monkeypatch.setattr("minifrontier.generation.sample_next_token", scripted_sample)
    result = model.generate(
        torch.tensor([[1, 2]]), max_new_tokens=6, stop_strings=["STOP"], decode=decode
    )
    # Stopped the step "STOP" completed (4 new tokens) -- never reached the
    # scripted 'x' tokens that would follow if generation kept going.
    assert result[:, 2:].tolist() == [[10, 11, 12, 13]]


def test_stop_strings_ends_generation_when_the_string_only_forms_across_a_token_boundary(
    monkeypatch,
) -> None:
    """Misaligned case: the stop string never appears within any single token's
    own decoded text, only once two multi-character tokens are concatenated --
    proves this re-decodes the growing suffix rather than checking the newest
    token in isolation."""

    model = MiniFrontier(ModelConfig.tiny_edu(vocab_size=32, max_seq_len=16)).eval()
    # 20='ST', 21='OP' -- "STOP" only exists once both are decoded together.
    scripted = [20, 21, 14, 14]
    calls = 0

    def scripted_sample(logits, **kwargs):
        nonlocal calls
        token = torch.tensor([[scripted[calls]]])
        calls += 1
        return token

    char_for = {20: "ST", 21: "OP", 14: "x"}

    def decode(ids):
        return "".join(char_for[i] for i in ids)

    monkeypatch.setattr("minifrontier.generation.sample_next_token", scripted_sample)
    result = model.generate(
        torch.tensor([[1, 2]]), max_new_tokens=4, stop_strings=["STOP"], decode=decode
    )
    assert result[:, 2:].tolist() == [[20, 21]]
