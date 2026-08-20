"""Versioned, opt-in execution scoring for original code fixtures (MF-036).

Beginner's map of this file
---------------------------
The only honest way to score generated code is to run it against tests. That also
means executing text a model just made up, so this path is **opt-in** rather than
automatic, and the fixtures are original to this project rather than borrowed from
a public benchmark -- public benchmark problems have long since leaked into
web-crawled training data, and a model that has read the answers is not being
measured, it is being flattered.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from minifrontier.shards import hamming_distance, simhash64


@dataclass(frozen=True, slots=True)
class CodeScore:
    syntax_valid: bool
    compiles: bool
    tests_passed: bool | None


def normalized_hash(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines()) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def score_python(
    source: str,
    *,
    tests: str | None = None,
    execute_trusted_fixture: bool = False,
    timeout_seconds: float = 2.0,
) -> CodeScore:
    """Parse/compile safely; execute only when the caller marks fixture code trusted."""

    try:
        ast.parse(source)
    except SyntaxError:
        return CodeScore(False, False, False if tests is not None else None)
    try:
        compile(source, "<candidate>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return CodeScore(True, False, False if tests is not None else None)
    if tests is None:
        return CodeScore(True, True, None)
    if not execute_trusted_fixture:
        raise ValueError("test execution is disabled; pass execute_trusted_fixture=True")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    program = f"{source}\n{tests}\n"
    with tempfile.TemporaryDirectory(prefix="minifrontier-eval-") as directory:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", program],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    return CodeScore(True, True, completed.returncode == 0)


def load_fixtures(path: str | Path) -> list[dict[str, object]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def assert_no_contamination(fixtures: list[dict[str, object]], training_hashes: set[str]) -> None:
    overlaps = {
        str(fixture["id"])
        for fixture in fixtures
        if normalized_hash(str(fixture["prompt"])) in training_hashes
        or normalized_hash(str(fixture["reference"])) in training_hashes
    }
    if overlaps:
        raise ValueError(f"evaluation/training contamination detected: {sorted(overlaps)}")


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    exact_fixture_ids: tuple[str, ...]
    near_fixture_ids: tuple[str, ...]
    max_hamming_distance: int

    @property
    def clean(self) -> bool:
        return not self.exact_fixture_ids and not self.near_fixture_ids


def contamination_report(
    fixtures: list[dict[str, object]],
    *,
    training_hashes: set[str] | frozenset[str] = frozenset(),
    training_simhashes: set[int] | frozenset[int] = frozenset(),
    max_hamming_distance: int = 3,
) -> ContaminationReport:
    exact: set[str] = set()
    near: set[str] = set()
    for fixture in fixtures:
        fixture_id = str(fixture["id"])
        texts = [str(fixture.get("prompt", "")), str(fixture.get("reference", ""))]
        if any(normalized_hash(text) in training_hashes for text in texts):
            exact.add(fixture_id)
        if any(
            hamming_distance(simhash64(text), signature) <= max_hamming_distance
            for text in texts
            for signature in training_simhashes
        ):
            near.add(fixture_id)
    return ContaminationReport(tuple(sorted(exact)), tuple(sorted(near)), max_hamming_distance)


@dataclass(frozen=True, slots=True)
class FixtureScore:
    fixture_id: str
    kind: str
    exact: bool
    syntax_valid: bool
    compiles: bool
    functional: bool | None


def score_fixture_predictions(
    fixtures: list[dict[str, object]],
    predictions: dict[str, str],
    *,
    execute_trusted_fixtures: bool = False,
) -> list[FixtureScore]:
    """Score versioned local fixtures without hiding missing predictions."""

    results = []
    for fixture in fixtures:
        fixture_id = str(fixture["id"])
        if fixture_id not in predictions:
            raise ValueError(f"missing prediction for fixture {fixture_id}")
        kind = str(fixture["kind"])
        prediction = predictions[fixture_id]
        reference = str(fixture["reference"])
        if kind == "fim":
            source = str(fixture["prompt"]) + prediction + str(fixture.get("suffix", ""))
        elif kind == "syntax_repair":
            source = prediction
        else:
            source = str(fixture["prompt"]) + prediction
        score = score_python(
            source,
            tests=str(fixture["tests"]) if fixture.get("tests") else None,
            execute_trusted_fixture=execute_trusted_fixtures,
        )
        results.append(
            FixtureScore(
                fixture_id,
                kind,
                prediction == reference,
                score.syntax_valid,
                score.compiles,
                score.tests_passed,
            )
        )
    return results
