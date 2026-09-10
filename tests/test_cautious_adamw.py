import pytest
import torch

from minifrontier.cautious_adamw import CautiousAdamW


def test_rejects_invalid_hyperparameters() -> None:
    param = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="lr"):
        CautiousAdamW([param], lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        CautiousAdamW([param], lr=-1.0)
    with pytest.raises(ValueError, match="betas"):
        CautiousAdamW([param], betas=(1.0, 0.999))
    with pytest.raises(ValueError, match="betas"):
        CautiousAdamW([param], betas=(0.9, -0.1))
    with pytest.raises(ValueError, match="eps"):
        CautiousAdamW([param], eps=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        CautiousAdamW([param], weight_decay=-0.1)
    with pytest.raises(ValueError, match="xi"):
        CautiousAdamW([param], xi=0.0)


def _reference_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    *,
    lr: float,
    betas: tuple,
    eps: float,
    weight_decay: float,
    xi: float,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Independent re-derivation of the paper's Algorithm 2, to cross-check the
    implementation rather than merely re-asserting its own arithmetic."""

    beta1, beta2 = betas
    step = state["step"] + 1
    exp_avg = beta1 * state["exp_avg"] + (1 - beta1) * grad
    exp_avg_sq = beta2 * state["exp_avg_sq"] + (1 - beta2) * grad * grad
    moment_hat = exp_avg / (1 - beta1**step)
    variance_hat = exp_avg_sq / (1 - beta2**step)
    update = moment_hat / (variance_hat.sqrt() + eps)
    mask = (update * grad > 0).float()
    scale = update.numel() / (mask.sum() + xi)
    effective_lr = lr * scale
    new_param = param - effective_lr * mask * update
    if weight_decay:
        new_param = new_param - effective_lr * weight_decay * new_param
    return new_param, {"step": step, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}, mask


def test_first_step_matches_the_hand_derived_formula_and_masks_nothing() -> None:
    # Step 1's momentum is exactly (1 - beta1) * grad -- always the same sign as
    # the fresh gradient -- so nothing can be masked out on the very first step.
    start = torch.tensor([1.0, 1.0])
    param = torch.nn.Parameter(start.clone())
    optimizer = CautiousAdamW([param], lr=0.1, weight_decay=0.0)
    grad = torch.tensor([10.0, -4.0])
    param.grad = grad.clone()
    optimizer.step()

    expected_param, _, expected_mask = _reference_step(
        start,
        grad,
        {"step": 0, "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        xi=1.0,
    )
    assert torch.equal(expected_mask, torch.ones(2))
    assert torch.allclose(param.detach(), expected_param, atol=1e-6)


def test_two_steps_match_the_hand_derived_formula_including_weight_decay() -> None:
    start = torch.tensor([1.0, 1.0])
    param = torch.nn.Parameter(start.clone())
    optimizer = CautiousAdamW([param], lr=0.1, weight_decay=0.2)
    grads = [torch.tensor([10.0, 10.0]), torch.tensor([10.0, -0.001])]

    reference_param = start
    reference_state = {"step": 0, "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)}
    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()
        reference_param, reference_state, _ = _reference_step(
            reference_param,
            grad,
            reference_state,
            lr=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.2,
            xi=1.0,
        )
        assert torch.allclose(param.detach(), reference_param, atol=1e-6)


def test_an_element_whose_momentum_disagrees_with_a_sign_flipped_gradient_is_masked() -> None:
    """Element 0: steady positive gradient both steps, never masked. Element 1: a
    large positive gradient first, then a tiny negative one -- accumulated
    momentum stays positive on step 2 even though the fresh gradient flipped
    sign, so this element's *update* term is masked out on step 2 (with
    weight_decay=0, meaning it must be exactly unchanged that step)."""

    param = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    optimizer = CautiousAdamW([param], lr=0.1, weight_decay=0.0)
    param.grad = torch.tensor([10.0, 10.0])
    optimizer.step()
    before = param.detach().clone()

    param.grad = torch.tensor([10.0, -0.001])
    optimizer.step()
    after = param.detach().clone()

    assert after[0] != before[0]
    assert after[1] == before[1]


def test_step_requires_grad_and_is_a_noop_for_parameters_without_one() -> None:
    with_grad = torch.nn.Parameter(torch.tensor([1.0]))
    without_grad = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = CautiousAdamW([with_grad, without_grad], lr=0.1)
    with_grad.grad = torch.tensor([5.0])
    optimizer.step()
    assert without_grad.item() == 2.0
    assert with_grad.item() != 1.0
