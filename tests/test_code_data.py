from __future__ import annotations

from minifrontier.code_data import (
    FIM_TRANSFORM_VERSION,
    CodeAdmissionStats,
    deterministic_fim,
    filter_code_documents,
    mix_fim_documents,
)
from minifrontier.data import Document, split_bucket
from minifrontier.evaluation.code import (
    contamination_report,
    load_fixtures,
    normalized_hash,
    score_fixture_predictions,
)
from minifrontier.shards import simhash64


def code_document(text: str, record_id: str, *, path: str = "src/module.py") -> Document:
    return Document.create(
        text,
        source="https://example.invalid/repository",
        revision="0123456789abcdef",
        license="Apache-2.0",
        language="Python",
        record_id=record_id,
        path=path,
        source_type="code",
    )


def test_code_admission_filters_sensitive_generated_and_vendor_content() -> None:
    good = code_document("def add(a, b):\n    return a + b\n# documented implementation", "good")
    secret = code_document("api_key = 'super-secret-value-123'\nprint(api_key)", "secret")
    generated = code_document("# AUTO-GENERATED - DO NOT EDIT\nvalue = 1\n", "generated")
    vendor = code_document("def vendored():\n    return True\n", "vendor", path="vendor/x.py")
    personal = code_document("owner = 'person@example.com'\nvalue = 1\n", "pii")
    stats = CodeAdmissionStats()
    admitted = list(filter_code_documents([good, secret, generated, vendor, personal], stats=stats))
    assert admitted == [good]
    assert stats.admitted == 1
    assert stats.reasons == {
        "secret_or_credential": 1,
        "generated": 1,
        "vendor_or_generated_path": 1,
        "personal_data": 1,
    }


def test_fim_transform_is_deterministic_reconstructable_and_versioned() -> None:
    source = "def clamp(value, low, high):\n    return max(low, min(value, high))\n"
    first = deterministic_fim(source, seed=7, identity="repo:path:hash")
    second = deterministic_fim(source, seed=7, identity="repo:path:hash")
    assert first == second
    assert first.reconstruct() == source
    assert first.version == FIM_TRANSFORM_VERSION
    assert first.render().startswith("<|fim_prefix|>")
    assert "<|fim_suffix|>" in first.render() and "<|fim_middle|>" in first.render()


def test_fim_mixing_changes_only_selected_code_and_preserves_provenance() -> None:
    code = code_document("def square(value):\n    return value * value\n", "one")
    transformed = next(iter(mix_fim_documents([code], rate=1.0, seed=11)))
    unchanged = next(iter(mix_fim_documents([code], rate=0.0, seed=11)))
    assert transformed.source == code.source
    assert transformed.revision == code.revision
    assert transformed.license == code.license
    assert transformed.content_hash != code.content_hash
    assert transformed.parent_content_hash == code.content_hash
    assert transformed.transform == FIM_TRANSFORM_VERSION
    assert split_bucket(transformed) == split_bucket(code)
    assert unchanged == code


def test_fim_default_rate_is_frozen_fifteen_percent() -> None:
    documents = [
        code_document(f"def value_{index}():\n    return {123456789 + index}\n", str(index))
        for index in range(200)
    ]
    transformed = list(mix_fim_documents(documents, seed=19))
    count = sum(":fim:" in document.record_id for document in transformed)
    assert 20 <= count <= 40


def test_code_fixture_report_exact_near_and_functional_scores() -> None:
    fixtures = load_fixtures("eval/fixtures/code_fim_v1.jsonl")
    predictions = {str(item["id"]): str(item["reference"]) for item in fixtures}
    scores = score_fixture_predictions(fixtures, predictions, execute_trusted_fixtures=True)
    assert all(score.exact and score.compiles and score.functional for score in scores)
    prompt = str(fixtures[0]["prompt"])
    report = contamination_report(
        fixtures,
        training_hashes={normalized_hash(prompt)},
        training_simhashes={simhash64(str(fixtures[1]["reference"]))},
    )
    assert str(fixtures[0]["id"]) in report.exact_fixture_ids
    assert str(fixtures[1]["id"]) in report.near_fixture_ids
    assert not report.clean
