"""Show what Multi-Token Prediction's auxiliary heads actually grade.

What this lab shows
--------------------
Normal training gives every position exactly one training signal: "was your
guess about the very next token right?" Multi-Token Prediction (MTP) adds
extra, smaller heads that also grade guesses further ahead -- "was your guess
about two tokens from now right? Three?" -- each contributing its own loss
term, summed with the main one.

The intuition: forcing a position to also predict further ahead means the
model has to commit to a sharper idea of where the text is going sooner,
rather than only ever being graded on the very next token. DeepSeek-V3 used
this in production at 671B/14.8T-token scale; the original paper (Gloeckle et
al., "Better & Faster Large Language Models via Multi-token Prediction") found
the benefit shrinks -- sometimes reverses -- at much smaller scale, which is
why this project treats it as an off-by-default experiment
(``scripts/compare_mtp.py``) rather than a default, and why the literature's
gains at massive scale should not be assumed to transfer here without testing.

This project's MTP is a deliberately simplified variant: each extra head is
one plain linear projection reading the same final hidden state the main
``lm_head`` reads (see ``minifrontier/mtp.py``'s own docstring for why, and why
MTP heads live entirely outside the model class and its checkpoint format).

Run it with::

    uv run --extra cpu python labs/08_mtp.py

What to look for. The three offsets grade three different, overlapping spans
of the same short sequence -- watch how the valid-target count shrinks as the
offset grows (there are fewer positions with something *k* steps ahead of
them), and how each head's loss starts high (its weights are freshly random,
so it is no better than guessing) before a single gradient step nudges it
down.
"""

import torch

from minifrontier.loss import next_token_loss_stats
from minifrontier.mtp import MTPHeads


def main() -> None:
    torch.manual_seed(21)
    d_model, vocab_size = 8, 32
    sequence_length = 10

    # Stand-in for a real model's final hidden states -- normally this is
    # exactly the tensor `MiniFrontier.forward(..., return_hidden_states=True)`
    # exposes, right before the main `lm_head` sees it.
    hidden_states = torch.randn(1, sequence_length, d_model, requires_grad=True)
    tokens = torch.randint(0, vocab_size, (1, sequence_length))

    print("sequence length:", sequence_length)
    print("tokens:", tokens.tolist())

    # The main head's loss, offset=1, exactly as every position is always
    # graded -- included here only so the extra heads' losses have something
    # to be compared against.
    main_logits = torch.randn(1, sequence_length, vocab_size, requires_grad=True)
    main_loss_sum, main_count = next_token_loss_stats(main_logits, tokens, offset=1)
    print(
        f"\nmain head   (offset=1): {int(main_count)} valid targets, "
        f"loss/token={main_loss_sum.item() / main_count.item():.3f}"
    )

    mtp = MTPHeads(d_model=d_model, vocab_size=vocab_size, n_extra_heads=2)
    for extra_index, head in enumerate(mtp.heads):
        offset = 2 + extra_index
        with torch.no_grad():
            logits = head(hidden_states)
        loss_sum, count = next_token_loss_stats(logits, tokens, offset=offset)
        print(
            f"extra head  (offset={offset}): {int(count)} valid targets, "
            f"loss/token={loss_sum.item() / count.item():.3f}"
        )

    # One combined gradient step, the way `training.py` actually uses this:
    # the auxiliary loss is summed (not averaged) across heads, weighted, and
    # added to the primary loss before a single division and backward call.
    loss_sum, count = mtp.loss_sum_and_count(hidden_states, tokens)
    weight = 0.3
    combined = main_loss_sum + weight * loss_sum
    combined.backward()
    print(
        f"\ncombined MTP auxiliary loss (both extra heads, weight={weight}): "
        f"{(loss_sum / count).item():.3f} per token"
    )
    print("hidden_states.grad is populated:", hidden_states.grad is not None)
    print("Real training uses train_updates(..., mtp_heads=...); this file is for understanding.")


if __name__ == "__main__":
    main()
