"""Compare a Modern forward with QK-Norm enabled and disabled.

What this lab shows
-------------------
QK-Norm is a volume limiter on the question and the name tag.

The problem it solves. As models get bigger, the Q-dot-K scores sometimes grow
enormous. Softmax then saturates: one token gets 100% of the attention and
everything else gets 0%. Gradients vanish through that layer, and the training
loss spikes into nonsense -- often thousands of GPU-hours into a run. It is one of
the classic ways a large training job dies.

The fix. Put an ``RMSNorm`` on Q and on K -- the same volume knob used everywhere
else, but applied per head, over just ``head_dim`` numbers. Before two tokens
compare cards, both cards get resized to a standard size, so the comparison is
about *direction* (do these match?) rather than about who shouted louder.

Order matters and is easy to get backwards on a whiteboard: in this codebase
QK-Norm runs **before** RoPE. Project, normalize, rotate.

Run it with::

    uv run --extra cpu python labs/03_qk_norm.py

What to look for. Two identically seeded Modern models, one with the flag on and
one off, on the same tokens, printing loss and total gradient norm for each.

Read this one carefully: on a randomly initialized toy model the two columns land
in the same neighbourhood, and neither ordering is evidence of anything. Untrained
weights are small, so the dot products QK-Norm exists to tame are not large in the
first place. What the lab actually demonstrates is that the flag is wired through
end to end and that both configurations train. The failure mode it prevents only
appears at scale, after many thousands of updates -- which is exactly why it is
worth understanding before you go looking for it in a loss curve.
"""

import torch

from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def main() -> None:
    # One seed for both models, so the random starting weights are comparable and
    # `qk_norm` really is the only variable being changed.
    torch.manual_seed(42)
    without = MiniFrontier(ModelConfig.tiny_modern(qk_norm=False, attention_impl="sdpa"))
    with_norm = MiniFrontier(ModelConfig.tiny_modern(qk_norm=True, attention_impl="sdpa"))
    tokens = torch.randint(0, without.config.vocab_size, (2, 16))
    for label, model in (("off", without), ("on", with_norm)):
        # labels=tokens is plain next-token prediction; see loss.py for the shift.
        loss = model(tokens, labels=tokens).loss
        assert loss is not None
        loss.backward()
        # clip_grad_norm_ with an infinite threshold clips nothing -- it is being
        # used purely as a convenient way to MEASURE the total gradient length.
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        print(f"qk_norm={label}: loss={loss.item():.6f}, grad_norm={gradient_norm.item():.6f}")


if __name__ == "__main__":
    main()
