"""Weighted language-model validation metrics (MF-034).

Beginner's map of this file
---------------------------
How well is the model doing on text it has never seen? Three views of the same
measurement:

* **Cross-entropy loss** (nats/token) -- average surprise. The training objective.
* **Perplexity** = ``exp(loss)`` -- read it as "the model is about as unsure as if
  it were picking uniformly among this many tokens". A perplexity of 20 means
  roughly a 1-in-20 guess. It is only comparable between models that share a
  tokenizer, because a different tokenizer changes what "one token" means.
* **Bits per byte (BPB)** -- the same information rescaled to raw UTF-8 bytes,
  which *is* comparable across tokenizers. That makes it the honest number when
  comparing against outside models.

"Weighted" is the reason this file exists rather than a one-line average: batches
contain different numbers of scored tokens, so the totals are summed and divided
once at the end. Averaging per-batch averages would quietly over-weight the small
batches.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from minifrontier.model import MiniFrontier
from minifrontier.tokenizer import MiniFrontierTokenizer


@dataclass(frozen=True, slots=True)
class ValidationBatch:
    """One packed token batch and its unpadded source-byte count."""

    tokens: torch.Tensor
    utf8_bytes: int


@dataclass(frozen=True, slots=True)
class LanguageMetrics:
    cross_entropy: float
    perplexity: float
    bits_per_byte: float
    predicted_tokens: int
    utf8_bytes: int


class MetricAccumulator:
    """Accumulate on-device totals and synchronize only when metrics are computed."""

    def __init__(self, device: torch.device) -> None:
        self.negative_log_likelihood = torch.zeros((), dtype=torch.float64, device=device)
        self.predicted_tokens = torch.zeros((), dtype=torch.int64, device=device)
        self.utf8_bytes = 0

    def update(self, nll: torch.Tensor, count: torch.Tensor, utf8_bytes: int) -> None:
        if utf8_bytes < 0:
            raise ValueError("utf8_bytes cannot be negative")
        self.negative_log_likelihood += nll.detach().to(torch.float64)
        self.predicted_tokens += count.detach().to(torch.int64)
        self.utf8_bytes += utf8_bytes

    def compute(self) -> LanguageMetrics:
        # This is the deliberate metric-collection synchronization boundary.
        total_nll = float(self.negative_log_likelihood.cpu())
        token_count = int(self.predicted_tokens.cpu())
        if token_count == 0:
            raise ValueError("validation contains no predicted, non-padding tokens")
        if self.utf8_bytes == 0:
            raise ValueError("validation contains no UTF-8 bytes")
        cross_entropy = total_nll / token_count
        return LanguageMetrics(
            cross_entropy=cross_entropy,
            perplexity=math.exp(cross_entropy),
            bits_per_byte=total_nll / (math.log(2.0) * self.utf8_bytes),
            predicted_tokens=token_count,
            utf8_bytes=self.utf8_bytes,
        )


@torch.inference_mode()
def evaluate_token_batches(
    model: MiniFrontier,
    batches: Iterable[ValidationBatch],
    *,
    pad_id: int,
) -> LanguageMetrics:
    """Evaluate shifted next-token loss without partial-batch or padding bias."""

    was_training = model.training
    model.eval()
    accumulator: MetricAccumulator | None = None
    try:
        for batch in batches:
            tokens = batch.tokens
            if tokens.ndim != 2 or tokens.shape[1] < 2:
                raise ValueError("validation tokens must have shape [batch, sequence>=2]")
            logits = model(tokens).logits[:, :-1].float()
            labels = tokens[:, 1:]
            valid = labels.ne(pad_id)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels)
            if accumulator is None:
                accumulator = MetricAccumulator(tokens.device)
            accumulator.update((losses * valid).sum(), valid.sum(), batch.utf8_bytes)
    finally:
        model.train(was_training)
    if accumulator is None:
        raise ValueError("validation batches cannot be empty")
    return accumulator.compute()


def batches_from_texts(
    tokenizer: MiniFrontierTokenizer,
    texts: Iterable[str],
    *,
    max_seq_len: int,
    device: torch.device | str = "cpu",
) -> list[ValidationBatch]:
    """Create unpadded per-document batches while retaining actual UTF-8 counts."""

    batches = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) > max_seq_len:
            raise ValueError(
                "validation document exceeds max_seq_len; split it before evaluation so "
                "token losses and UTF-8 bytes remain aligned"
            )
        batches.append(
            ValidationBatch(
                tokens=torch.tensor([ids], dtype=torch.long, device=device),
                utf8_bytes=len(text.encode("utf-8")),
            )
        )
    return batches
