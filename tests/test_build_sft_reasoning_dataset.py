"""MF-123: tests for scripts/build_sft_reasoning_dataset.py.

Uses a small, fake in-memory dataset (monkeypatched over the real `load_dataset`
call) rather than the real GSM8K corpus -- fast, fully offline, and independent
of whether this machine happens to have the real dataset cached.
"""

from __future__ import annotations

import json

import pytest

import scripts.build_sft_reasoning_dataset as build_sft
from minifrontier.sft import iter_conversations

_FAKE_TRAIN = [
    {"question": "What is 2 + 2?", "answer": "2 + 2 = <<2+2=4>>4.\n#### 4"},
    {"question": "What is 3 + 5?", "answer": "3 + 5 = <<3+5=8>>8.\n#### 8"},
]
_FAKE_TEST_DISJOINT = [
    {"question": "What is 10 - 1?", "answer": "10 - 1 = <<10-1=9>>9.\n#### 9"},
]
_FAKE_TEST_OVERLAPPING = [
    {"question": "  What  is 2 + 2?  ", "answer": "irrelevant -- same question as train"},
]


def _fake_load_dataset(test_rows: list[dict[str, str]]):
    def load(_name: str, _config: str, *, split: str):
        return _FAKE_TRAIN if split == "train" else test_rows

    return load


def test_normalized_question_collapses_whitespace_and_case() -> None:
    assert build_sft._normalized_question("  What Is 2 + 2?  ") == "what is 2 + 2?"
    assert build_sft._normalized_question("what is 2 + 2?") == "what is 2 + 2?"


def test_check_no_train_test_overlap_is_zero_for_real_disjoint_splits(monkeypatch) -> None:
    monkeypatch.setattr(build_sft, "load_dataset", _fake_load_dataset(_FAKE_TEST_DISJOINT))
    assert build_sft.check_no_train_test_overlap() == 0


def test_check_no_train_test_overlap_actually_detects_a_real_overlap(monkeypatch) -> None:
    """Proves the check would catch real contamination, not just report 0 by
    construction -- a test question that only differs by whitespace/case from
    a train question must still be flagged."""

    monkeypatch.setattr(build_sft, "load_dataset", _fake_load_dataset(_FAKE_TEST_OVERLAPPING))
    assert build_sft.check_no_train_test_overlap() == 1


def test_build_records_produces_correct_provenance_and_content(monkeypatch) -> None:
    monkeypatch.setattr(build_sft, "load_dataset", _fake_load_dataset(_FAKE_TEST_DISJOINT))
    records = build_sft.build_records(limit=None)
    assert len(records) == 2
    first = records[0]
    assert first.source == build_sft.SOURCE_DATASET
    assert first.revision == build_sft.SOURCE_REVISION
    assert first.license == build_sft.SOURCE_LICENSE
    assert first.record_id == "gsm8k-train-00000"
    assert first.messages[0].role == "user"
    assert first.messages[0].content == "What is 2 + 2?"
    assert first.messages[1].role == "assistant"
    assert first.messages[1].content == "2 + 2 = <<2+2=4>>4.\n#### 4"


def test_build_records_respects_limit(monkeypatch) -> None:
    monkeypatch.setattr(build_sft, "load_dataset", _fake_load_dataset(_FAKE_TEST_DISJOINT))
    assert len(build_sft.build_records(limit=1)) == 1


def test_main_refuses_to_write_when_contamination_is_found(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(build_sft, "load_dataset", _fake_load_dataset(_FAKE_TEST_OVERLAPPING))
    output_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        "sys.argv", ["build_sft_reasoning_dataset.py", "--output", str(output_path)]
    )
    with pytest.raises(ValueError, match="contamination"):
        build_sft.main()
    assert not output_path.exists()


def test_main_writes_a_real_record_that_round_trips_through_sft_py(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(build_sft, "load_dataset", _fake_load_dataset(_FAKE_TEST_DISJOINT))
    output_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        "sys.argv", ["build_sft_reasoning_dataset.py", "--output", str(output_path)]
    )
    build_sft.main()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["messages"] == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = <<2+2=4>>4.\n#### 4"},
    ]

    records = list(iter_conversations(output_path))
    assert len(records) == 2
    assert records[0].record_id == "gsm8k-train-00000"
