"""Exponential moving average of model weights (MF-086, part 4).

Beginner's map of this file
---------------------------
A single training run's final weights are one noisy sample from a loss surface
that keeps jittering update to update. An EMA keeps a second, "shadow" copy of
every weight that only ever moves a small fraction of the way toward the live
weights on each update -- so it tracks the same overall trajectory but averages
out the jitter. Swapping the shadow weights in at evaluation/release time is a
well-known, nearly-free way to get a slightly better (and less noisy) model
than the raw trained weights, at the cost of one extra copy of the parameters
in memory.

This lives outside ``MiniFrontier`` itself (like ``mtp.py``'s heads) so the
frozen architecture and checkpoint format are untouched: EMA is purely a
training-time bookkeeping structure, opt-in via ``TrainingConfig.ema_decay``.
"""

from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file

from minifrontier.model import MiniFrontier


class EMAWeights:
    """Tracks a decaying average of a model's parameters, deduplicating tied weights.

    ``decay`` close to 1 (e.g. 0.999) barely moves the shadow each update
    (long memory, very smooth); closer to 0 tracks the live weights almost
    exactly (short memory, little averaging). Tied parameters (e.g. tied
    embeddings) are stored once, matching ``training.py``'s own
    ``_parameter_groups`` deduplication-by-``id()`` pattern -- otherwise the
    same physical tensor would be updated twice per step under two different
    names, silently double-applying the decay.
    """

    def __init__(self, model: MiniFrontier, *, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        seen: set[int] = set()
        for name, parameter in model.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            self._shadow[name] = parameter.detach().clone()

    @torch.no_grad()
    def update(self, model: MiniFrontier) -> None:
        """Move the shadow a `1 - decay` fraction of the way toward the live weights."""

        seen: set[int] = set()
        for name, parameter in model.named_parameters():
            if id(parameter) in seen or name not in self._shadow:
                continue
            seen.add(id(parameter))
            shadow = self._shadow[name]
            shadow.mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)

    def copy_to(self, model: MiniFrontier) -> None:
        """Overwrite `model`'s live parameters with the current shadow weights in place."""

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self._shadow:
                    parameter.copy_(self._shadow[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._shadow)

    def save(self, path: str) -> None:
        save_file(self._shadow, path)

    def load(self, path: str) -> None:
        loaded = load_file(path)
        if set(loaded) != set(self._shadow):
            raise ValueError("EMA checkpoint parameter names do not match this model")
        for name, tensor in loaded.items():
            self._shadow[name] = tensor.clone()
