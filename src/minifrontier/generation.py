"""KV-cached greedy, temperature, top-k, and nucleus generation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from minifrontier.cache import KVCache

if TYPE_CHECKING:
    from minifrontier.model import MiniFrontier


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
    generator: torch.Generator | None = None,
    validate_logits: bool = False,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("sampling logits must be [batch, vocab]")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if validate_logits and not torch.isfinite(logits).all():
        raise ValueError("sampling logits contain non-finite values")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    filtered = logits.float() / temperature
    if top_k is not None and top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, top_k, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: MiniFrontier,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    validate_logits: bool = False,
) -> torch.Tensor:
    if prompt.ndim != 2 or prompt.shape[1] == 0:
        raise ValueError("prompt must be non-empty [batch, sequence] tokens")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus requested tokens exceeds model max_seq_len")
    if eos_id is not None and not 0 <= eos_id < model.config.vocab_size:
        raise ValueError("eos_id is outside the model vocabulary")
    if max_new_tokens == 0:
        return prompt.clone()

    was_training = model.training
    model.eval()
    try:
        cache = KVCache.allocate(
            model.config,
            batch_size=prompt.shape[0],
            device=prompt.device,
            dtype=None,
            capacity=prompt.shape[1] + max_new_tokens,
            bounded_local=model.config.attention_pattern == "hybrid",
        )
        output = torch.empty(
            (prompt.shape[0], prompt.shape[1] + max_new_tokens),
            dtype=prompt.dtype,
            device=prompt.device,
        )
        output[:, : prompt.shape[1]].copy_(prompt)
        output_length = prompt.shape[1]
        logits = model(prompt, cache=cache, logits_to_keep=1).logits[:, 0, :]
        finished = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
        for step in range(max_new_tokens):
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
                validate_logits=validate_logits,
            )
            if eos_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
                finished |= next_token.squeeze(1).eq(eos_id)
            output[:, output_length : output_length + 1].copy_(next_token)
            output_length += 1
            if step + 1 == max_new_tokens or finished.all():
                break
            logits = model(next_token, cache=cache, logits_to_keep=1).logits[:, 0, :]
        return output[:, :output_length]
    finally:
        model.train(was_training)
