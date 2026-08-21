"""Show how RoPE's rotation angle grows with position, and that it encodes distance.

What this lab shows
--------------------
RoPE rotates each token's Query and Key vectors by an angle proportional to that
token's position. Two things are worth seeing directly rather than taking on
faith:

1. Each of a head's ``head_dim / 2`` feature pairs ("arrows") rotates at its own
   fixed speed, and that angle grows linearly with position -- arrow 0 fastest,
   the last arrow slowest.
2. The dot product between a rotated Query and a rotated Key -- the number
   attention actually compares -- depends only on the *distance* between their
   positions, not on where in the sequence they sit. Query at position 5 and Key
   at position 8 produce the same score as Query at position 50 and Key at
   position 53: both are "3 apart".

Run it with::

    uv run --extra cpu python labs/01_rope.py

What to look for. The angle table grows linearly down each column, fastest arrow
first. Then, for a fixed distance of 3 tokens, the Q-K dot product is printed for
several different starting positions -- those numbers should all match. Changing
the distance changes the number; changing only the starting position does not.
That is the entire relative-position guarantee RoPE is built to provide.
"""

import torch

from minifrontier.rope import RoPE, apply_rotary


def main() -> None:
    head_dim = 8
    rope = RoPE(head_dim=head_dim, max_seq_len=64)
    print(f"head_dim={head_dim} gives {head_dim // 2} rotating feature pairs (arrows).")
    print("Rotation speed per arrow (radians/token):\n", rope.inverse_frequency)

    # Fact 1: angle = position * speed, and it grows linearly with position. Print
    # the angle of arrow 0 (fastest) and the last arrow (slowest) at a few
    # positions so the growth -- and the difference in speed -- is visible.
    positions = torch.tensor([0, 1, 2, 5, 10, 20])
    cosine, sine = rope(positions, dtype=torch.float32, device=torch.device("cpu"))
    angles = torch.atan2(sine, cosine)  # recover the angle itself from cos/sin
    print("(angles come from atan2, so they wrap into (-pi, pi] once position*speed exceeds pi)")
    print("\nPosition | fastest-arrow angle | slowest-arrow angle")
    slowest = head_dim // 2 - 1
    for row, position in enumerate(positions.tolist()):
        fastest_angle = angles[row, 0].item()
        slowest_angle = angles[row, slowest].item()
        print(f"{position:>8} | {fastest_angle:>19.4f} | {slowest_angle:>19.4f}")
    print("Each column grows linearly in position -- that is 'twist token 5 more than token 2'.")

    # Fact 2: relative-position invariance. Fix one Query and one Key vector, then
    # rotate them at several (start, start + distance) position pairs sharing the
    # same distance. The dot product between the rotated vectors should be
    # constant across every start, because it depends only on the distance.
    torch.manual_seed(0)
    query = torch.randn(1, 1, 1, head_dim)
    key = torch.randn(1, 1, 1, head_dim)
    distance = 3
    print(f"\nFixed distance={distance}: Q@K dot product across different starting positions")
    for start in (0, 1, 5, 20, 40):
        query_position = torch.tensor([start])
        key_position = torch.tensor([start + distance])
        q_cos, q_sin = rope(query_position, dtype=torch.float32, device=torch.device("cpu"))
        k_cos, k_sin = rope(key_position, dtype=torch.float32, device=torch.device("cpu"))
        rotated_query = apply_rotary(query, q_cos, q_sin)
        rotated_key = apply_rotary(key, k_cos, k_sin)
        dot_product = (rotated_query * rotated_key).sum().item()
        print(f"  Query@{start:<3} Key@{start + distance:<3} -> dot product = {dot_product:.6f}")

    print("\nNow vary the distance itself -- the dot product should change:")
    for distance in (0, 1, 3, 10):
        query_position = torch.tensor([7])
        key_position = torch.tensor([7 + distance])
        q_cos, q_sin = rope(query_position, dtype=torch.float32, device=torch.device("cpu"))
        k_cos, k_sin = rope(key_position, dtype=torch.float32, device=torch.device("cpu"))
        rotated_query = apply_rotary(query, q_cos, q_sin)
        rotated_key = apply_rotary(key, k_cos, k_sin)
        dot_product = (rotated_query * rotated_key).sum().item()
        print(f"  distance={distance:<3} -> dot product = {dot_product:.6f}")

    print(
        "\nThe first block holds distance fixed and the dot product barely moves "
        "(floating point only); the second block changes distance and the dot "
        "product changes with it. That is RoPE encoding relative position."
    )


if __name__ == "__main__":
    main()
