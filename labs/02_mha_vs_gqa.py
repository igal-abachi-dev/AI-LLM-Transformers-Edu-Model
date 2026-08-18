"""Measure the parameter and KV-cache consequences of MHA versus GQA."""

import torch

from minifrontier.cache import KVCache
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def main() -> None:
    edu_config = ModelConfig.tiny_edu(n_layers=4)
    modern_config = ModelConfig.tiny_modern(n_layers=4, attention_impl="sdpa")
    edu = MiniFrontier(edu_config)
    modern = MiniFrontier(modern_config)
    edu_cache = KVCache.allocate(edu_config, batch_size=1, device="cpu", dtype=torch.float32)
    modern_cache = KVCache.allocate(modern_config, batch_size=1, device="cpu", dtype=torch.float32)
    print(f"MHA parameters: {edu.parameter_count():,}")
    print(f"GQA parameters: {modern.parameter_count():,}")
    print(f"MHA cache bytes: {edu_cache.allocated_bytes():,}")
    print(f"GQA cache bytes: {modern_cache.allocated_bytes():,}")


if __name__ == "__main__":
    main()
