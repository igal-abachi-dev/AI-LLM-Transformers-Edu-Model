"""Small transparent scoring helpers for versioned SFT prompt fixtures.

Beginner's map of this file
---------------------------
After fine-tuning, the questions are simpler than benchmark accuracy: does the
model answer instead of continuing the question, does it stop cleanly at
``<|eos|>``, does it stay inside the chat format? These deliberately shallow,
readable checks measure exactly that and claim nothing more.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def score_sft_responses(
    prompts: Sequence[Mapping[str, object]],
    responses: Mapping[str, str],
) -> dict[str, object]:
    rows = []
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        response = responses.get(prompt_id, "")
        required = [str(value) for value in prompt.get("required_substrings", [])]
        matched = all(value in response for value in required)
        rows.append(
            {
                "id": prompt_id,
                "category": str(prompt["category"]),
                "non_empty": bool(response.strip()),
                "required_substrings": required,
                "required_match": matched,
                "response": response,
            }
        )
    count = max(len(rows), 1)
    return {
        "count": len(rows),
        "non_empty_rate": sum(bool(row["non_empty"]) for row in rows) / count,
        "required_match_rate": sum(bool(row["required_match"]) for row in rows) / count,
        "rows": rows,
    }
