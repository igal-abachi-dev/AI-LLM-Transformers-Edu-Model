from pathlib import Path

import pytest

import scripts.compare_tokenizers as compare_tokenizers


def test_parse_arm_spec_splits_on_semicolon_not_colon() -> None:
    # Windows drive-letter paths ("C:\...") rule out ":" as the delimiter --
    # this is the whole reason the format uses ";".
    spec = "16k;C:\\configs\\a.toml;C:\\data\\tok;C:\\data\\train;C:\\data\\val"
    arm = compare_tokenizers._parse_arm_spec(spec)
    assert arm.label == "16k"
    assert arm.config_path == Path("C:\\configs\\a.toml")
    assert arm.tokenizer_path == Path("C:\\data\\tok")
    assert arm.train_shards == Path("C:\\data\\train")
    assert arm.validation_shards == Path("C:\\data\\val")


def test_parse_arm_spec_rejects_wrong_field_count() -> None:
    with pytest.raises(ValueError, match="got 4 field"):
        compare_tokenizers._parse_arm_spec("label;config;tokenizer;train")


def test_parse_arm_spec_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="label must be non-empty"):
        compare_tokenizers._parse_arm_spec(";config;tokenizer;train;validation")


def test_parse_args_rejects_duplicate_arm_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_tokenizers.py",
            "--arm",
            "a;c1;t1;tr1;v1",
            "--arm",
            "a;c2;t2;tr2;v2",
            "--output",
            "out",
            "--seconds",
            "60",
        ],
    )
    with pytest.raises(ValueError, match="unique"):
        compare_tokenizers.parse_args()
