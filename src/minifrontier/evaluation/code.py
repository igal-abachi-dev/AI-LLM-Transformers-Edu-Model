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
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
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


_CSHARP_PROJECT = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
"""


def dotnet_available() -> bool:
    return shutil.which("dotnet") is not None


def score_csharp(source: str, *, timeout_seconds: float = 60.0) -> CodeScore:
    """Real `dotnet build` compile check for a standalone C# fixture.

    Deliberately narrower than `score_python`: a real compile check only, no
    functional test execution -- wiring a C# test runner is a real, separate
    piece of scope this project has not built. `syntax_valid` and `compiles`
    report the same real boolean, since `dotnet build` conflates parsing and
    compilation in one step and this project does not want a direct Roslyn
    library dependency just to separate them the way Python's `ast.parse`
    versus `compile` naturally does.

    ``source`` must be valid as *members of a class* (methods, properties,
    nested types) -- it is wrapped as ``class Fixture { <source> }`` and
    built with ``OutputType=Library`` (no entry point required). This
    deliberately sidesteps two real, verified C# rules that make "top-level
    statements" the wrong shape for a fixture-scoring harness: an `Exe`
    project requires a real entry point (bare declarations alone do not
    satisfy it, `CS5001`), while top-level statements themselves are only
    legal in an `Exe` project and must precede any type declaration in the
    same file (`CS8805`/`CS8803`) -- both confirmed by direct, real
    `dotnet build` attempts before this design was settled on, not assumed.
    No external NuGet package references are used, so a real build never
    needs network access once the SDK itself is installed locally.
    """

    if not dotnet_available():
        raise RuntimeError("dotnet SDK not found on PATH; C# scoring requires it")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with tempfile.TemporaryDirectory(prefix="minifrontier-eval-cs-") as directory:
        project_dir = Path(directory)
        (project_dir / "fixture.csproj").write_text(_CSHARP_PROJECT, encoding="utf-8")
        (project_dir / "Program.cs").write_text(
            f"class Fixture {{\n{source}\n}}\n", encoding="utf-8"
        )
        env = {
            **os.environ,
            "DOTNET_NOLOGO": "1",
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        }
        try:
            completed = subprocess.run(
                ["dotnet", "build", "--nologo", "-v", "quiet"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return CodeScore(False, False, None)
    compiled = completed.returncode == 0
    return CodeScore(compiled, compiled, None)


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


def _score_one_prediction(
    fixture: dict[str, object],
    fixture_id: str,
    prediction: str,
    *,
    execute_trusted_fixtures: bool,
) -> CodeScore:
    """Build the real candidate source for one fixture/prediction pair and score it.

    Shared by the single-sample path (`score_fixture_predictions`) and the
    multi-sample pass@k path (`score_fixture_predictions_pass_at_k`, MF-148) so
    the two never drift on how a candidate is assembled or dispatched.
    """

    kind = str(fixture["kind"])
    language = str(fixture.get("language", "python"))
    if kind == "fim":
        source = str(fixture["prompt"]) + prediction + str(fixture.get("suffix", ""))
    elif kind == "syntax_repair":
        source = prediction
    else:
        source = str(fixture["prompt"]) + prediction
    if language == "python":
        return score_python(
            source,
            tests=str(fixture["tests"]) if fixture.get("tests") else None,
            execute_trusted_fixture=execute_trusted_fixtures,
        )
    if language == "csharp":
        if fixture.get("tests"):
            raise ValueError(f"fixture {fixture_id}: C# scoring does not support tests yet")
        return score_csharp(source)
    raise ValueError(f"fixture {fixture_id}: unknown language {language!r}")


def score_fixture_predictions(
    fixtures: list[dict[str, object]],
    predictions: dict[str, str],
    *,
    execute_trusted_fixtures: bool = False,
) -> list[FixtureScore]:
    """Score versioned local fixtures without hiding missing predictions.

    One prediction per fixture -- a real, honest pass/fail, but noisy on any
    fixture a model can pass by luck on one attempt without passing reliably.
    See `score_fixture_predictions_pass_at_k` (MF-148) for the multi-sample,
    noise-aware alternative modeled on the real HumanEval/Codex methodology.
    """

    results = []
    for fixture in fixtures:
        fixture_id = str(fixture["id"])
        if fixture_id not in predictions:
            raise ValueError(f"missing prediction for fixture {fixture_id}")
        prediction = predictions[fixture_id]
        reference = str(fixture["reference"])
        score = _score_one_prediction(
            fixture, fixture_id, prediction, execute_trusted_fixtures=execute_trusted_fixtures
        )
        results.append(
            FixtureScore(
                fixture_id,
                str(fixture["kind"]),
                prediction == reference,
                score.syntax_valid,
                score.compiles,
                score.tests_passed,
            )
        )
    return results


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased pass@k estimator from the real HumanEval/Codex paper.

    (Chen et al., "Evaluating Large Language Models Trained on Code",
    arXiv:2107.03374, equation 1.) Given ``n`` independent samples for one
    problem, of which ``c`` pass, this is the probability that *at least one*
    of ``k`` samples drawn (without replacement) from those ``n`` would pass --
    not simply ``c / n`` (which is pass@n, a different, noisier number) and
    not "run k samples and see if any pass" (a biased estimator with higher
    variance than this closed form for the same n).

    Uses the numerically stable product form
    ``1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)`` rather than computing
    ``comb(n-c, k) / comb(n, k)`` directly, which real reference
    implementations (including OpenAI's own) avoid because those binomial
    coefficients overflow for even moderately large ``n``.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= c <= n:
        raise ValueError("c must be in [0, n]")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        raise ValueError(f"k={k} cannot exceed n={n} samples")
    if n - c < k:
        # Fewer than k samples failed, so at least one of any k drawn must pass.
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@dataclass(frozen=True, slots=True)
class FixturePassAtK:
    fixture_id: str
    kind: str
    num_samples: int
    num_functional_passes: int
    pass_at_k: dict[int, float]


def score_fixture_predictions_pass_at_k(
    fixtures: list[dict[str, object]],
    predictions: dict[str, Sequence[str]],
    *,
    k_values: Sequence[int],
    execute_trusted_fixtures: bool = False,
) -> list[FixturePassAtK]:
    """Multi-sample pass@k scoring (MF-148) -- the real HumanEval/Codex
    methodology, generating several candidate completions per fixture and
    reporting the unbiased pass@k estimator rather than a single-sample
    pass/fail. This exists specifically because a single sample cannot tell
    "reliably correct" apart from "got lucky once" -- see `pass_at_k`.

    Every fixture must supply at least ``max(k_values)`` samples; fixtures
    that can only be sampled once (e.g. a deterministic reference-predictions
    smoke) should keep using `score_fixture_predictions` instead, which this
    function does not replace.

    Only fixtures with real ``tests`` are meaningful here -- a fixture with no
    tests has no notion of "functional pass" to estimate pass@k over, so one
    without tests raises rather than silently reporting a meaningless number.
    """

    if not k_values:
        raise ValueError("at least one k value is required")
    if any(k <= 0 for k in k_values):
        raise ValueError("k values must be positive")
    max_k = max(k_values)
    results = []
    for fixture in fixtures:
        fixture_id = str(fixture["id"])
        if not fixture.get("tests"):
            raise ValueError(f"fixture {fixture_id} has no tests; pass@k requires functional tests")
        if fixture_id not in predictions:
            raise ValueError(f"missing predictions for fixture {fixture_id}")
        samples = predictions[fixture_id]
        if len(samples) < max_k:
            raise ValueError(
                f"fixture {fixture_id} has {len(samples)} samples, fewer than the "
                f"requested max k={max_k}"
            )
        num_functional_passes = sum(
            1
            for prediction in samples
            if _score_one_prediction(
                fixture,
                fixture_id,
                prediction,
                execute_trusted_fixtures=execute_trusted_fixtures,
            ).tests_passed
        )
        n = len(samples)
        results.append(
            FixturePassAtK(
                fixture_id,
                str(fixture["kind"]),
                n,
                num_functional_passes,
                {k: pass_at_k(n, num_functional_passes, k) for k in k_values},
            )
        )
    return results
