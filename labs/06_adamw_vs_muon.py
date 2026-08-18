"""Inspect the educational Newton-Schulz step used to explain first-party Muon."""

import torch

from minifrontier.muon import newton_schulz_reference


def main() -> None:
    torch.manual_seed(55)
    gradient = torch.randn(4, 6)
    orthogonalized = newton_schulz_reference(gradient)
    print("input shape:", tuple(gradient.shape))
    print("input Frobenius norm:", gradient.norm().item())
    print("Muon reference singular values:", torch.linalg.svdvals(orthogonalized).tolist())
    print("Production training uses torch.optim.Muon, never this teaching function.")


if __name__ == "__main__":
    main()
