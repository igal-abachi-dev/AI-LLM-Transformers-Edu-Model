"""KV-cache lab placeholder; scheduled after MF-029.

Status: placeholder
-------------------
Not written yet; running it exits with a message. It is meant to show cached and
uncached generation producing the same tokens, and to plot how cache memory grows
with conversation length -- linearly for global layers, and not at all for a local
layer's ring buffer once its window is full.

Until then:

* ``src/minifrontier/cache.py`` -- the linear and ring caches, both commented.
* ``tests/test_cache.py`` and ``tests/test_generation.py`` -- including the parity
  check that cached and uncached logits agree.
* ``introduction.md`` section 2.8 -- why an old token's Key and Value never change,
  which is the entire justification for caching them.
"""


def main() -> None:
    raise SystemExit("KV-cache lab requires MF-029")


if __name__ == "__main__":
    main()
