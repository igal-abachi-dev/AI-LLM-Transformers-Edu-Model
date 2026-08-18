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
