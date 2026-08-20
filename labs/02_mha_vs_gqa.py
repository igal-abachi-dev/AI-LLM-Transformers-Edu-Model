"""Measure the parameter and KV-cache consequences of MHA versus GQA.

What this lab shows
-------------------
GQA -- grouped-query attention -- is the single biggest lever on what it costs to
*serve* a model, and this lab puts real numbers on it.

The problem it solves. During generation, every layer stores a Key and a Value for
every token so far (see ``cache.py``). For a 150M Edu model at 2,048 tokens that
is roughly 126 MB of cache per conversation. Serve fifty people at once and you
have spent 6 GB on notes alone. At production scale -- 80 layers, 32k context --
the cache, not the weights, is what limits how many users fit on a GPU.

The observation. The *queries* need to be diverse: that is the model asking
different questions. But four separate name tags and four separate envelopes turn
out to be mostly redundant.

The fix. Keep every query head; use fewer key/value heads and let query heads
share them. Here 4 query heads share 2 KV heads, so the cache halves. The real
150M Modern preset uses 12 query heads over 4 KV heads, cutting it by three.

Run it with::

    uv run --extra cpu python labs/02_mha_vs_gqa.py

What to look for. The parameter counts move a little -- Modern is about 7%
smaller here, since ``k_proj``/``v_proj`` are half as wide -- but the **cache
bytes** halve exactly, 32,768 to 16,384. That contrast is the whole point:
a modest change in model size, a proportional change in serving cost. It is why
GQA appears in essentially every model card since Llama 2 70B.
"""

import torch

from minifrontier.cache import KVCache
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def main() -> None:
    # Same depth on both sides so the only difference that matters is the head
    # layout: Edu is 4 query / 4 KV heads (MHA), Modern is 4 query / 2 KV (GQA).
    edu_config = ModelConfig.tiny_edu(n_layers=4)
    modern_config = ModelConfig.tiny_modern(n_layers=4, attention_impl="sdpa")
    edu = MiniFrontier(edu_config)
    modern = MiniFrontier(modern_config)
    # Caches sized for a full-length conversation. Note `bounded_local` is left off
    # here, so this measures the GQA saving alone -- lab 04 covers the hybrid
    # schedule's separate, and larger, saving on top of it.
    edu_cache = KVCache.allocate(edu_config, batch_size=1, device="cpu", dtype=torch.float32)
    modern_cache = KVCache.allocate(modern_config, batch_size=1, device="cpu", dtype=torch.float32)
    print(f"MHA parameters: {edu.parameter_count():,}")
    print(f"GQA parameters: {modern.parameter_count():,}")
    print(f"MHA cache bytes: {edu_cache.allocated_bytes():,}")
    print(f"GQA cache bytes: {modern_cache.allocated_bytes():,}")


if __name__ == "__main__":
    main()
