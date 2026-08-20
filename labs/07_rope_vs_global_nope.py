"""Isolate the global RoPE-versus-NoPE flag on identical random weights.

What this lab shows
-------------------
NoPE means "no position stamp at all" on the global layers -- they see an
unordered pile of tokens. That sounds broken, and it is genuinely an open question
in the field, which is why this repository labels it an experiment rather than a
feature.

The idea: the local layers underneath have already baked ordering into the
residual stream, so a global layer can lean on that. And a rotation that was only
ever trained up to 2,048 positions may actively hurt when you later push the model
to much longer contexts -- so not having one might extrapolate better.

Run it with::

    uv run --extra cpu python labs/07_rope_vs_global_nope.py

What to look for. This lab proves **flag isolation**, not quality. Two models are
built, the second is given the first's exact weights, and the same tokens go
through both. A non-zero maximum logit difference means the flag really did change
the computation and only the global layers' position handling; if it printed 0.0,
the switch would be silently doing nothing.

That is a worthwhile thing to check on its own. Whether NoPE is *better* can only
be answered by matched training runs, and this script deliberately makes no such
claim -- see the last line it prints.
"""

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
    # Copy every weight across, so the two models differ ONLY in the position flag.
    # Without this they would also differ in their random initialization, and the
    # comparison would mean nothing.
    nope.load_state_dict(rope.state_dict())
    tokens = torch.randint(0, rope.config.vocab_size, (1, 16))
    # Largest disagreement anywhere in the output scoreboard.
    difference = (rope(tokens).logits - nope(tokens).logits).abs().max().item()
    print(f"maximum logit difference: {difference:.6f}")
    print("This random-weight check proves flag isolation, not a NoPE quality conclusion.")


if __name__ == "__main__":
    main()
