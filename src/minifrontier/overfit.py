"""Bounded CPU/GPU overfit proof for M0/M1 correctness."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.reproducibility import seed_everything
from minifrontier.run_metadata import RunMetadata


@dataclass(slots=True)
class OverfitResult:
    initial_loss: float
    final_loss: float
    steps: int
    generated: list[int]


def pattern_batch(
    examples: int,
    sequence_length: int,
    vocab_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if examples <= 0 or sequence_length < 2 or vocab_size < 4:
        raise ValueError("examples > 0, sequence_length >= 2, and vocab_size >= 4 are required")
    starts = torch.arange(examples, device=device).unsqueeze(1) % (vocab_size - 1)
    offsets = torch.arange(sequence_length, device=device).unsqueeze(0)
    return ((starts + offsets) % (vocab_size - 1) + 1).long()


def run_overfit(
    config: ModelConfig,
    *,
    examples: int = 100,
    sequence_length: int = 12,
    steps: int = 700,
    learning_rate: float = 3e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[MiniFrontier, OverfitResult, RunMetadata]:
    if sequence_length > config.max_seq_len:
        raise ValueError("sequence_length exceeds the model context")
    seed_everything(seed)
    torch_device = torch.device(device)
    model = MiniFrontier(config).to(torch_device)
    tokens = pattern_batch(examples, sequence_length, config.vocab_size, device=torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    started = time.perf_counter()
    initial_loss = 0.0
    final_loss = 0.0
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens).loss
        assert loss is not None
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = loss.item()
    elapsed = time.perf_counter() - started
    prompt = tokens[:1, :2]
    generated = model.generate(prompt, max_new_tokens=4).squeeze(0).tolist()
    trained_tokens = examples * (sequence_length - 1) * steps
    metadata = RunMetadata(
        name=f"{config.preset}-overfit",
        config=config.to_dict(),
        seed=seed,
        parameters=model.parameter_count(),
        train_tokens=trained_tokens,
        wall_seconds=elapsed,
        tokens_per_second=trained_tokens / elapsed,
        train_loss=final_loss,
        metrics={"initial_loss": initial_loss},
    )
    result = OverfitResult(initial_loss, final_loss, steps, generated)
    return model, result, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional TOML config; defaults to tiny Edu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/overfit/run.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ModelConfig.from_toml(args.config) if args.config else ModelConfig.tiny_edu()
    _, result, metadata = run_overfit(
        config,
        examples=args.examples,
        sequence_length=args.sequence_length,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    metadata.write_json(args.output)
    print(f"loss: {result.initial_loss:.4f} -> {result.final_loss:.4f}")
    print(f"generated token IDs: {result.generated}")
    print(f"run record: {args.output}")


if __name__ == "__main__":
    main()
