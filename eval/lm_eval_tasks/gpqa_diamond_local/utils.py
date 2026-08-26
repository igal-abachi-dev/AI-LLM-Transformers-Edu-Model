"""doc_to_target for the locally-provided GPQA-Diamond parquet.

The file at ``gpqa-diamond/test/gpqa_diamond.parquet`` (downloaded directly from
Hugging Face by the project owner, who has accepted the dataset's gated terms)
has only two columns: ``question`` (a self-contained prompt already ending in a
shuffled "A. ... B. ... C. ... D. ..." choice block) and ``answer`` (a single
correct-choice letter). This differs from the raw ``Idavidrein/gpqa`` schema
lm-eval's own ``gpqa_diamond_zeroshot`` task expects (``Question``,
``Correct Answer``, ``Incorrect Answer 1/2/3``) -- this task works with the
already-shuffled two-column form actually on disk instead.
"""

from __future__ import annotations


def doc_to_target(doc: dict[str, str]) -> int:
    return ord(str(doc["answer"]).strip().upper()) - ord("A")
