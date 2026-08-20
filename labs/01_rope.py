"""RoPE visualization lab placeholder.

Status: placeholder
-------------------
Not written yet; running it exits with a message. The plan is to plot how the
rotation angle of each feature pair grows with position, and show that the dot
product between two rotated vectors depends only on the distance between them.

Until then:

* ``src/minifrontier/rope.py`` -- 60-odd commented lines covering the whole idea.
* ``tests/test_rope.py`` -- checks the split-half convention against an
  independent implementation, which is the bug this design is most prone to.
* ``introduction.md`` section 2.2 -- the plain-language version: give each token's
  numbers a twist, and twist token 5 more than token 2.
"""


def main() -> None:
    raise SystemExit("RoPE lab is not implemented yet")


if __name__ == "__main__":
    main()
