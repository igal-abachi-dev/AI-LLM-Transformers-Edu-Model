"""Isolate the global RoPE-versus-NoPE flag on identical random weights."""

import torch

from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def main() -> None:
    torch.manual_seed(42)
    rope = MiniFrontier(
        ModelConfig.tiny_modern(global_position_encoding="rope", attention_impl="sdpa")
    )
    nope = MiniFrontier(
        ModelConfig.tiny_modern(global_position_encoding="none", attention_impl="sdpa")
    )
    nope.load_state_dict(rope.state_dict())
    tokens = torch.randint(0, rope.config.vocab_size, (1, 16))
    difference = (rope(tokens).logits - nope(tokens).logits).abs().max().item()
    print(f"maximum logit difference: {difference:.6f}")
    print("This random-weight check proves flag isolation, not a NoPE quality conclusion.")


if __name__ == "__main__":
    main()
