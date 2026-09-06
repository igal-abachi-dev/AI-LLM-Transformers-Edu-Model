"""See why the final projection layer, not the transformer body, breaks memory.

What this lab shows
--------------------
Every position's hidden state gets projected to a score for every vocabulary
token before cross-entropy can be computed: ``[B, S, d_model] -> [B, S,
vocab_size]``. At a small ``d_model`` and a not-small ``vocab_size``, that
tensor -- plus the FP32 gradient the loss needs during backward -- is
disproportionately large next to the rest of the model.

The arithmetic below is exact, not approximate: element counts times dtype
size. Two numbers matter:

* **Naive** -- project the whole sequence at once, keep the full logits (and
  their gradient) in memory simultaneously. This is what
  ``next_token_loss``/``next_token_loss_stats`` do, and it is fine as long as
  ``batch_tokens * vocab_size`` stays small relative to available VRAM.
* **Chunked** -- ``loss.chunked_next_token_loss_stats`` projects and scores a
  few hundred positions at a time, discarding each chunk's logits before
  building the next. Peak memory becomes one chunk's logits, not the whole
  sequence's -- see that function's own docstring, and ``_ChunkedCrossEntropy``
  above it, for why this needs a hand-derived backward rather than just a loop
  (a plain loop would still need one ``.backward()`` per chunk, and each of
  those re-walks the *entire* transformer to reach ``hidden``'s parents).

Run it with::

    uv run --extra cpu python labs/09_chunked_vs_fused_cross_entropy.py

What to look for. The "naive resident" row is what a single microbatch holds
in memory *at once* under the old path; the "chunked resident" row is what it
holds under the new one, regardless of how long the sequence is. If CUDA is
available, the lab also runs both paths for real and prints measured peak
allocated memory, not just the arithmetic estimate.

Why this project doesn't use a fused Triton kernel ("Cut Cross-Entropy",
arXiv:2411.09009) instead: `AGENTS.md`'s frozen V1 scope excludes custom
CUDA/Triton kernels. The chunked, hand-derived-backward approach here gets
the overwhelming majority of a fused kernel's memory benefit (peak memory
drops to one chunk either way) using only plain PyTorch, at the cost of a
small, measured amount of VRAM (roughly one chunk's worth) that a fused
kernel could additionally avoid. See MF-084 in `tasks/backlog.md` for the
full reasoning and the FLOPs accounting behind why a *naive* (recompute-based)
chunking approach was rejected in favor of this one.
"""

from __future__ import annotations

import torch

from minifrontier.loss import chunked_next_token_loss_stats, next_token_loss_stats

BYTES_PER_FP32_ELEMENT = 4


def _describe_bytes(n_bytes: int) -> str:
    return f"{n_bytes:,} bytes ({n_bytes / 1e6:.1f} MB)"


def print_arithmetic(
    *, batch_size: int, sequence_length: int, vocab_size: int, chunk_size: int
) -> None:
    batch_tokens = batch_size * sequence_length
    full_elements = batch_tokens * vocab_size
    chunk_elements = min(chunk_size, batch_tokens) * vocab_size
    # Naive keeps logits AND their FP32 gradient alive simultaneously during
    # backward; chunked never holds more than one chunk's logits (its gradient
    # is computed and discarded within the same chunk iteration -- see
    # _ChunkedCrossEntropy's docstring).
    naive_bytes = full_elements * BYTES_PER_FP32_ELEMENT * 2
    chunked_bytes = chunk_elements * BYTES_PER_FP32_ELEMENT * 2

    print(f"\nvocab_size={vocab_size}, batch_tokens={batch_tokens}, chunk_size={chunk_size}")
    print(f"  full [batch_tokens, vocab] logits: {full_elements:,} elements")
    print(f"  naive resident (logits + grad):    {_describe_bytes(naive_bytes)}")
    print(f"  chunked resident (one chunk only): {_describe_bytes(chunked_bytes)}")
    print(f"  reduction: {naive_bytes / chunked_bytes:.1f}x")


def measure_on_cuda(*, batch_tokens: int, d_model: int, vocab_size: int, chunk_size: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    tokens = torch.randint(0, vocab_size, (1, batch_tokens + 1), device=device)

    def run(fn: callable) -> int:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        hidden = torch.randn(1, batch_tokens + 1, d_model, device=device, requires_grad=True)
        weight = torch.randn(vocab_size, d_model, device=device, requires_grad=True)
        fn(hidden, weight)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device)

    def naive(hidden: torch.Tensor, weight: torch.Tensor) -> None:
        logits = hidden @ weight.T
        loss_sum, count = next_token_loss_stats(logits, tokens)
        (loss_sum / count).backward()

    def chunked(hidden: torch.Tensor, weight: torch.Tensor) -> None:
        loss_sum, count, _ = chunked_next_token_loss_stats(
            hidden, weight, tokens, chunk_size=chunk_size
        )
        (loss_sum / count).backward()

    naive_peak = run(naive)
    chunked_peak = run(chunked)
    print(f"\nMeasured on {torch.cuda.get_device_name(device)}:")
    print(f"  naive peak allocated:   {_describe_bytes(naive_peak)}")
    print(f"  chunked peak allocated: {_describe_bytes(chunked_peak)}")
    print(f"  reduction: {naive_peak / chunked_peak:.1f}x")


def main() -> None:
    print("Element-count arithmetic (always accurate, no GPU needed):")
    print_arithmetic(batch_size=2, sequence_length=1024, vocab_size=16_384, chunk_size=512)
    print_arithmetic(batch_size=2, sequence_length=1024, vocab_size=32_768, chunk_size=512)

    if torch.cuda.is_available():
        measure_on_cuda(batch_tokens=2048, d_model=768, vocab_size=32_768, chunk_size=512)
    else:
        print("\nNo CUDA device available -- skipping the real measured comparison.")
        print("The arithmetic above is exact regardless; only the real peak-allocation")
        print("numbers (allocator overhead, fragmentation) need actual hardware.")


if __name__ == "__main__":
    main()
