"""Show exact full/local mask density without making a speed or quality claim."""

from minifrontier.masking import build_attention_mask


def main() -> None:
    sequence = 2_048
    window = 512
    full = build_attention_mask(sequence, sequence)
    local = build_attention_mask(sequence, sequence, window_size=window)
    print(f"full allowed pairs: {int(full.sum()):,}")
    print(f"local allowed pairs: {int(local.sum()):,}")
    print(f"pair ratio: {local.sum().item() / full.sum().item():.4f}")
    print("This is a semantics/cost count, not a fused-kernel speed measurement.")


if __name__ == "__main__":
    main()
