"""Manual attention derivation lab; use tests/test_attention.py until expanded.

Status: placeholder
-------------------
This lab is scheduled but not written yet, so running it exits with a message
rather than pretending to teach something.

Until it lands, the same ground is covered by:

* ``src/minifrontier/attention.py`` -- read
  ``manual_scaled_dot_product_attention`` first. It is
  ``softmax(Q @ K^T / sqrt(head_dim) + mask) @ V`` spelled out line by line, and
  every line is commented.
* ``tests/test_attention.py`` -- short, and it shows what each piece is supposed
  to do, including the check that the readable version and PyTorch's fused kernel
  agree.
* ``introduction.md`` section 2.4 -- the same idea explained with a classroom of
  children holding queries, name tags, and envelopes.
"""


def main() -> None:
    raise SystemExit("interactive lab is scheduled after the tested core is frozen")


if __name__ == "__main__":
    main()
