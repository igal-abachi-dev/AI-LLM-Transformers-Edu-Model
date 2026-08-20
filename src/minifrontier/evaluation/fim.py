"""Fill-in-the-middle reference scoring (MF-036/MF-054).

Beginner's map of this file
---------------------------
Given a hole in a file, did the model produce the missing piece? Scored by exact
match against the reference middle, which is strict but unambiguous -- and
unambiguous matters more than generous when the number goes into a release note.
See ``code_data.py`` for how FIM examples are built in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

from minifrontier.evaluation.code import CodeScore, score_python


@dataclass(frozen=True, slots=True)
class FIMScore:
    exact: bool
    functional: bool | None
    code: CodeScore


def score_fim(
    prefix: str,
    suffix: str,
    predicted_middle: str,
    reference_middle: str,
    *,
    tests: str | None = None,
    execute_trusted_fixture: bool = False,
) -> FIMScore:
    candidate = prefix + predicted_middle + suffix
    code_score = score_python(
        candidate,
        tests=tests,
        execute_trusted_fixture=execute_trusted_fixture,
    )
    return FIMScore(
        exact=predicted_middle == reference_middle,
        functional=code_score.tests_passed,
        code=code_score,
    )
