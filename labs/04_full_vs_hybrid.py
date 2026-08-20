"""Show exact full/local mask density without making a speed or quality claim.

What this lab shows
-------------------
Full attention means every token looks at every earlier token. Double the context
and you quadruple the work -- that quadratic wall is why long context is
expensive.

The observation behind hybrid attention: most of language is local. To finish
"the cat sat on the ___" you need the last six words, not paragraph three. Only
occasionally is long reach needed -- a variable declared 400 lines up, a name from
the top of the document.

So don't make every layer far-sighted. Make most of them near-sighted and cheap,
and put a proper long-range layer in every fourth position. Information still
travels: layer 0 mixes each token with its neighbours, layer 1 mixes those
already-mixed tokens with theirs, and the reach compounds -- like a rumour
crossing a classroom one desk at a time. Every fourth layer, someone shouts.

This lab counts the allowed (query, key) pairs in each mask, which is the exact
semantic cost -- how many comparisons the maths permits.

Run it with::

    uv run --extra cpu python labs/04_full_vs_hybrid.py

What to look for. At 2,048 tokens with a 512 window, local allows about 44% of
full's pairs. Two things to take away: that gap widens fast as context grows, and
the KV-cache saving is larger still, because a local layer only ever has to
*remember* 512 tokens no matter how long the conversation runs.

Note the honesty of the last printed line. This is a count of permitted pairs, not
a wall-clock measurement -- a fused kernel's real speed depends on hardware and
tiling, and claiming otherwise from this script would be wrong.
"""

from minifrontier.masking import build_attention_mask


def main() -> None:
    sequence = 2_048
    window = 512
    # Same function, one extra argument. `full` is the causal triangle; `local` is
    # that triangle narrowed to a diagonal band of `window` cells.
    full = build_attention_mask(sequence, sequence)
    local = build_attention_mask(sequence, sequence, window_size=window)
    print(f"full allowed pairs: {int(full.sum()):,}")
    print(f"local allowed pairs: {int(local.sum()):,}")
    print(f"pair ratio: {local.sum().item() / full.sum().item():.4f}")
    print("This is a semantics/cost count, not a fused-kernel speed measurement.")


if __name__ == "__main__":
    main()
