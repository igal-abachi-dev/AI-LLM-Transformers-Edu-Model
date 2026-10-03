"""See what LoRA actually buys: a real parameter-count cut, and a real zero-init check.

What this lab shows
--------------------
Full fine-tuning updates every weight in a linear layer. LoRA (Hu et al.,
arXiv:2106.09685, ``src/minifrontier/lora.py``) freezes the original weight
entirely and trains a small, separate rank-``r`` update alongside it instead:

    output = base(x) + scale * B @ (A @ x)

The problem it solves. An optimizer like AdamW keeps two extra running
averages per trainable weight, so a full fine-tune of even this project's own
150M model needs real memory for 150M-worth of gradients plus 300M-worth of
optimizer state, on top of the weights themselves. LoRA's two small matrices
per wrapped layer (``rank x in_features`` and ``out_features x rank``) are
the only things an optimizer ever has to carry state for.

The one detail that makes it work correctly, not just efficiently. ``A``
starts at a small random value, but ``B`` starts at exactly zero -- so a
freshly wrapped layer's forward pass is numerically *identical* to the
unwrapped base layer, not merely close. This lab verifies that directly,
the same way this project verifies everything else (real numbers, not just
a description of the idea).

Run it with::

    uv run --extra cpu python labs/11_lora.py

What to look for. The zero-init check should read ``0.00e+00`` -- an exact
match. Then the real parameter count: wrapping only the attention
projections (q/k/v/out) with a small rank leaves well under half of this
tiny model's parameters trainable, and the gap only grows at real scale,
since the frozen base weight count grows with ``in_features * out_features``
while the adapter cost grows with only ``rank * (in_features + out_features)``.
"""

import torch

from minifrontier.config import ModelConfig
from minifrontier.lora import (
    apply_lora,
    mark_only_lora_trainable,
    trainable_parameter_counts,
)
from minifrontier.model import MiniFrontier


def main() -> None:
    torch.manual_seed(0)
    model = MiniFrontier(ModelConfig.tiny_edu(n_layers=4, d_model=64, n_heads=4, d_ff=192))
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))

    # Zero-init check: a forward pass BEFORE wrapping, saved for comparison.
    before = model(tokens).logits.clone()

    rank = 4
    wrapped_names = apply_lora(
        model, target_pattern=r"\.(q_proj|k_proj|v_proj|out_proj)$", rank=rank
    )
    mark_only_lora_trainable(model)

    after = model(tokens).logits
    max_diff = (after - before).abs().max().item()
    print(f"Wrapped {len(wrapped_names)} layers at rank={rank}: {wrapped_names}")
    print(f"Max output difference immediately after wrapping: {max_diff:.2e} (should be 0.00e+00)")

    trainable, total = trainable_parameter_counts(model)
    print(
        f"\nTrainable parameters: {trainable:,} of {total:,} total "
        f"({trainable / total:.1%}) -- the rest stays frozen at full precision."
    )

    # One real training step, to show the adapter actually learns while the
    # base weights genuinely never move.
    base_weight_before = model.blocks[0].attention.q_proj.base.weight.clone()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    loss = model(tokens, labels=tokens).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    base_weight_after = model.blocks[0].attention.q_proj.base.weight
    base_moved = not torch.equal(base_weight_before, base_weight_after)
    print(
        f"\nAfter one real optimizer step, did the frozen base weight move? "
        f"{base_moved} (should be False)"
    )


if __name__ == "__main__":
    main()
