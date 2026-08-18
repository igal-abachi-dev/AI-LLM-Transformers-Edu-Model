"""Compare a Modern forward with QK-Norm enabled and disabled."""

import torch

from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def main() -> None:
    torch.manual_seed(42)
    without = MiniFrontier(ModelConfig.tiny_modern(qk_norm=False, attention_impl="sdpa"))
    with_norm = MiniFrontier(ModelConfig.tiny_modern(qk_norm=True, attention_impl="sdpa"))
    tokens = torch.randint(0, without.config.vocab_size, (2, 16))
    for label, model in (("off", without), ("on", with_norm)):
        loss = model(tokens, labels=tokens).loss
        assert loss is not None
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        print(f"qk_norm={label}: loss={loss.item():.6f}, grad_norm={gradient_norm.item():.6f}")


if __name__ == "__main__":
    main()
