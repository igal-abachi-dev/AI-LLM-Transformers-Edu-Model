"""Show cached decode matches uncached recomputation, and how cache memory grows.

What this lab shows
--------------------
An old token's Key and Value never change once computed, so caching them lets
generation read one new token's worth of work per step instead of re-running the
whole conversation through every layer each time.

Two things are worth seeing directly:

1. **Correctness.** Feeding the whole sequence at once (no cache) and feeding it
   one token at a time (with a cache) must produce identical logits at every
   position -- the cache is a memory optimization, not an approximation.
2. **Memory shape.** A global layer's linear cache grows with every token
   generated -- one more slot, forever. A Modern local layer's ring cache is
   capped at ``local_window`` slots: once the window is full, writing the next
   token overwrites the oldest one, so its *logical* memory (real tokens
   currently stored) stops growing entirely, however long the conversation runs.

Run it with::

    uv run --extra cpu python labs/05_kv_cache.py

What to look for. The "cached vs uncached" section reports zero max difference and
identical greedy tokens. The memory table shows the global layer's logical bytes
climbing by a fixed amount every step while the local layer's logical bytes rise
only until step ``local_window``, then hold flat -- printed side by side so the
plateau versus the ramp is obvious at a glance.
"""

import torch

from minifrontier.cache import KVCache
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def main() -> None:
    torch.manual_seed(7)
    # Modern's hybrid schedule: layers 0-2 are local (ring-eligible), layer 3 is
    # global. A 4-token window makes the ring wrap quickly enough to observe here.
    config = ModelConfig.tiny_modern(
        max_seq_len=32, n_layers=4, local_window=4, attention_impl="sdpa"
    )
    model = MiniFrontier(config).eval()
    sequence_length = 12
    tokens = torch.randint(0, config.vocab_size, (1, sequence_length))

    # --- Part 1: cached decode must match one uncached forward over everything ---
    full_logits = model(tokens).logits
    cache = KVCache.allocate(
        config, batch_size=1, device="cpu", capacity=sequence_length, bounded_local=True
    )
    cached_pieces = [
        model(tokens[:, index : index + 1], cache=cache).logits for index in range(sequence_length)
    ]
    cached_logits = torch.cat(cached_pieces, dim=1)
    max_difference = (full_logits - cached_logits).abs().max().item()
    same_tokens = torch.equal(full_logits.argmax(dim=-1), cached_logits.argmax(dim=-1))
    print(f"Uncached-vs-cached max logit difference: {max_difference:.3e}")
    print(f"Uncached-vs-cached greedy tokens identical: {same_tokens}")
    assert max_difference < 2e-4
    assert same_tokens

    # --- Part 2: watch memory as decoding proceeds, one layer of each kind ---
    local_layer_index = next(i for i in range(config.n_layers) if config.is_local_layer(i))
    global_layer_index = next(i for i in range(config.n_layers) if not config.is_local_layer(i))
    print(
        f"\nlocal_window={config.local_window}; layer {local_layer_index} is local (ring), "
        f"layer {global_layer_index} is global (linear)."
    )

    growth_cache = KVCache.allocate(
        config, batch_size=1, device="cpu", capacity=sequence_length, bounded_local=True
    )
    print("\nstep | local logical bytes | global logical bytes")
    for step in range(sequence_length):
        model(tokens[:, step : step + 1], cache=growth_cache)
        local_bytes = growth_cache.layers[local_layer_index].logical_bytes()
        global_bytes = growth_cache.layers[global_layer_index].logical_bytes()
        print(f"{step:>4} | {local_bytes:>20} | {global_bytes:>21}")

    print(
        f"\nThe local column stops growing after step {config.local_window - 1} (window full); "
        "the global column keeps climbing every step. Same conversation, two very "
        "different memory shapes -- that gap is the entire point of the hybrid schedule."
    )


if __name__ == "__main__":
    main()
