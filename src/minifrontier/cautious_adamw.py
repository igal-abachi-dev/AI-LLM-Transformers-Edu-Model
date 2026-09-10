"""Cautious AdamW (MF-083): mask AdamW's update by gradient-sign agreement.

Beginner's map of this file
---------------------------
Plain AdamW's momentum term can, on any given step, point a *different*
direction than the current raw gradient -- momentum is a running average of
recent gradients, so it lags behind a gradient that has just changed sign.
Applying that stale-direction update anyway is, for that one step, moving the
weight the "wrong" way relative to what the data just said.

Cautious AdamW (Liang, Chen, Liu, Liu, "Cautious Optimizers: Improving
Training with One Line of Code", arXiv:2411.16085, ICLR 2026) fixes exactly
this: on every parameter, elementwise, only apply the update where its sign
agrees with the current gradient's sign; leave disagreeing elements
untouched this step. Skipped elements are not lost forever -- the next
step's fresh gradient will likely push them again. To keep the *effective*
step size from shrinking as more elements get masked out, the learning rate
for that step is scaled up by how much of the tensor was actually masked
(``dim / (aligned_count + xi)``) -- this is also what lets the paper prove
the modification preserves AdamW's own convergence guarantee, rather than
being pure heuristic. Weight decay is applied afterward, unmasked, at that
same scaled rate -- caution protects the gradient signal, not the
regularizer.

This is a from-scratch, readable reference implementation (a manual
per-parameter loop, like ``muon.py``'s Newton-Schulz reference) rather than a
fused/foreach kernel -- correctness and legibility first, matching this
project's manual-attention-kept-alongside-SDPA precedent.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


class CautiousAdamW(torch.optim.Optimizer):
    """AdamW whose update is masked to only the elements agreeing with the gradient."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, object]],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        xi: float = 1.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if xi <= 0:
            raise ValueError("xi must be positive")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "xi": xi,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            xi = group["xi"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("CautiousAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                moment_hat = exp_avg / bias_correction1
                variance_hat = exp_avg_sq / bias_correction2
                update = moment_hat / (variance_hat.sqrt() + eps)
                # phi_t: 1 where this update element points the same way the
                # current (not momentum-smoothed) gradient does, 0 otherwise.
                mask = (update * gradient > 0).to(update.dtype)
                # Rescale so a heavily-masked tensor still takes a full-size
                # step on its surviving elements -- see module docstring.
                scale = update.numel() / (mask.sum() + xi)
                effective_lr = lr * scale
                parameter.sub_(effective_lr * mask * update)
                if weight_decay:
                    parameter.sub_(effective_lr * weight_decay * parameter)
        return loss
