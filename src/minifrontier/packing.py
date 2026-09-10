"""Alternative document-packing algorithms for token shards (MF-097).

Beginner's map of this file
---------------------------
The default packing scheme (``TokenShardWriter``'s ``"ribbon"`` mode) is the
simplest possible thing: concatenate every document's tokens into one long
stream, in the order documents arrive, and slice off fixed-size rows. It
wastes zero tokens, but it means most rows end with one document's tail
followed immediately by an unrelated document's head -- an artificial
boundary the model has to learn to ignore, and whichever document happens to
straddle a row boundary gets truncated arbitrarily.

Both algorithms here instead buffer a batch of whole documents (already
tokenized, one EOS-terminated token list per document) and decide, all at
once, how to group them into rows -- trading "stream one document at a
time" for "see many documents, then pack them well":

* ``pack_best_fit`` -- Best-Fit-Decreasing bin packing (Ding et al., "Fewer
  Truncations Improve Language Modeling", arXiv:2404.10830): sort documents
  longest-first, and slot each into whichever partially-filled row has the
  least room left that can still hold it. This keeps every document intact
  (only a document *longer than one row* is ever split, into full-length
  chunks -- unavoidable at a fixed sequence length regardless of packing
  strategy) while packing rows tighter than arrival order alone would.
  Zero tokens are ever discarded.

* ``pack_bos_aligned_crop`` -- nanochat's scheme: every row starts fresh
  with a BOS token, then best-fit-fills with more whole documents; if the
  next document does not fit and room remains, its head is cropped to fill
  the row exactly and the rest of that document is discarded for good. This
  guarantees every row's boundaries are clean (BOS, then only whole or
  cleanly-cropped documents) at a real, disclosed cost: real token volume is
  thrown away, unlike ``pack_best_fit``.

Both return ``[(row_tokens, non_padding_count), ...]``, the same
``(tokens, non_padding)`` shape ``TokenShardWriter`` already accumulates for
its default ribbon rows, so it takes either output ready to pad-checked and
flush.
"""

from __future__ import annotations


def _split_oversized(
    documents: list[list[int]], *, sequence_length: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Break any document longer than one row into full-length chunks.

    Returns ``(full_chunks, remainder_or_original)`` -- chunks already
    exactly ``sequence_length`` long (ready to emit as-is, no packing
    needed) separately from everything still eligible for bin packing
    (a not-originally-oversized document, unchanged, or the final leftover
    piece of one that was split).
    """

    full_chunks: list[list[int]] = []
    packable: list[list[int]] = []
    for document in documents:
        if len(document) <= sequence_length:
            packable.append(document)
            continue
        remaining = document
        while len(remaining) > sequence_length:
            full_chunks.append(remaining[:sequence_length])
            remaining = remaining[sequence_length:]
        if remaining:
            packable.append(remaining)
    return full_chunks, packable


def pack_best_fit(
    documents: list[list[int]], *, sequence_length: int, pad_id: int
) -> list[tuple[list[int], int]]:
    """Best-Fit-Decreasing bin packing. Never discards a token."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    full_chunks, packable = _split_oversized(documents, sequence_length=sequence_length)
    rows: list[list[int]] = []
    remaining_capacity: list[int] = []
    for document in sorted(packable, key=len, reverse=True):
        best_row = -1
        best_remaining = sequence_length + 1  # sentinel: worse than any real row
        for index, capacity in enumerate(remaining_capacity):
            if len(document) <= capacity < best_remaining:
                best_row = index
                best_remaining = capacity
        if best_row == -1:
            rows.append(list(document))
            remaining_capacity.append(sequence_length - len(document))
        else:
            rows[best_row].extend(document)
            remaining_capacity[best_row] -= len(document)
    packed = [
        (row + [pad_id] * capacity, sequence_length - capacity)
        for row, capacity in zip(rows, remaining_capacity, strict=True)
    ]
    return [(chunk, sequence_length) for chunk in full_chunks] + packed


def pack_bos_aligned_crop(
    documents: list[list[int]], *, sequence_length: int, pad_id: int, bos_id: int
) -> list[tuple[list[int], int]]:
    """nanochat-style BOS-aligned best-fit-and-crop. Discards overflow tokens."""

    if sequence_length < 2:
        raise ValueError("sequence_length must allow room for BOS plus content")
    packed: list[tuple[list[int], int]] = []
    pointer = 0
    while pointer < len(documents):
        row = [bos_id]
        remaining = sequence_length - 1
        while pointer < len(documents) and len(documents[pointer]) <= remaining:
            row.extend(documents[pointer])
            remaining -= len(documents[pointer])
            pointer += 1
        if pointer < len(documents) and remaining > 0:
            # Nothing left fits whole -- crop the next document's head to fill
            # this row exactly, and permanently discard the rest of it.
            row.extend(documents[pointer][:remaining])
            remaining = 0
            pointer += 1
        packed.append((row + [pad_id] * remaining, sequence_length - remaining))
    return packed
