import json
from pathlib import Path

import pytest
import yaml

import scripts.build_dataset_card as build_dataset_card


def _write_metadata(
    path: Path, *, train_tokens: int, validation_tokens: int, admitted: int
) -> None:
    metadata = {
        "admission": {"admitted": admitted, "seen": admitted + 5, "reasons": {}},
        "train": {"total_non_padding_tokens": train_tokens},
        "validation": {"total_non_padding_tokens": validation_tokens},
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_build_card_computes_real_totals_weights_and_size_category(tmp_path: Path) -> None:
    fineweb_metadata = tmp_path / "fineweb.json"
    code_metadata = tmp_path / "code.json"
    _write_metadata(fineweb_metadata, train_tokens=600_000, validation_tokens=6_000, admitted=100)
    _write_metadata(code_metadata, train_tokens=396_000, validation_tokens=4_000, admitted=50)

    entries = build_dataset_card._load_sources(
        [
            ["fineweb-edu", "0.6", str(fineweb_metadata)],
            ["github-code", "0.4", str(code_metadata)],
        ]
    )
    card = build_dataset_card.build_card("MiniFrontier Mixture", entries)

    assert "# MiniFrontier Mixture" in card
    assert "| fineweb-edu | 60% | 606,000 | 100 |" in card
    assert "| github-code | 40% | 400,000 | 50 |" in card
    assert "| **Total** | **100%** | **1,006,000** |" in card
    assert "license: other" in card
    assert "size_categories:\n  - 1M<n<10M" in card
    assert "check the row's own `license` and `source` columns" in card
    assert "  - config_name: fineweb-edu" in card
    assert "    data_dir: fineweb-edu" in card
    assert "  - config_name: github-code" in card
    assert "    data_dir: github-code" in card
    assert "├── fineweb-edu/" in card
    assert "│   ├── train-00000-of-00001.parquet" in card
    assert "│   └── validation-00000-of-00001.parquet" in card


def test_frontmatter_is_valid_yaml_with_the_real_hf_configs_schema(tmp_path: Path) -> None:
    fineweb_metadata = tmp_path / "fineweb.json"
    code_metadata = tmp_path / "code.json"
    _write_metadata(fineweb_metadata, train_tokens=600_000, validation_tokens=6_000, admitted=100)
    _write_metadata(code_metadata, train_tokens=396_000, validation_tokens=4_000, admitted=50)
    entries = build_dataset_card._load_sources(
        [
            ["fineweb-edu", "0.6", str(fineweb_metadata)],
            ["github-code", "0.4", str(code_metadata)],
        ]
    )

    card = build_dataset_card.build_card("MiniFrontier Mixture", entries)
    frontmatter = card.split("---")[1]
    parsed = yaml.safe_load(frontmatter)

    assert parsed["configs"] == [
        {"config_name": "fineweb-edu", "data_dir": "fineweb-edu"},
        {"config_name": "github-code", "data_dir": "github-code"},
    ]
    assert parsed["license"] == "other"
    assert parsed["size_categories"] == ["1M<n<10M"]


def test_load_sources_rejects_weights_not_summing_to_one(tmp_path: Path) -> None:
    metadata_path = tmp_path / "a.json"
    _write_metadata(metadata_path, train_tokens=10, validation_tokens=0, admitted=1)
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        build_dataset_card._load_sources([["a", "0.5", str(metadata_path)]])


def test_load_sources_rejects_out_of_range_weight(tmp_path: Path) -> None:
    metadata_path = tmp_path / "a.json"
    _write_metadata(metadata_path, train_tokens=10, validation_tokens=0, admitted=1)
    with pytest.raises(ValueError, match="weight must be in"):
        build_dataset_card._load_sources([["a", "1.5", str(metadata_path)]])
    with pytest.raises(ValueError, match="weight must be in"):
        build_dataset_card._load_sources([["a", "0.0", str(metadata_path)]])


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (500, "n<1K"),
        (5_000, "1K<n<10K"),
        (50_000, "10K<n<100K"),
        (500_000, "100K<n<1M"),
        (5_000_000, "1M<n<10M"),
        (50_000_000, "10M<n<100M"),
        (500_000_000, "100M<n<1B"),
        (3_000_000_000, "1B<n<10B"),  # the real MF-070 3B-token target
        (30_000_000_000, "10B<n<100B"),
        (500_000_000_000, "100B<n<1T"),
        (5_000_000_000_000, "n>1T"),
    ],
)
def test_size_category_boundaries(tokens: int, expected: str) -> None:
    assert build_dataset_card._size_category(tokens) == expected


def test_main_writes_card_end_to_end(monkeypatch, tmp_path: Path) -> None:
    metadata_path = tmp_path / "source.json"
    _write_metadata(metadata_path, train_tokens=100, validation_tokens=0, admitted=3)
    output_path = tmp_path / "card" / "README.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_dataset_card.py",
            "--source",
            "only-source",
            "1.0",
            str(metadata_path),
            "--dataset-name",
            "Test Mixture",
            "--output",
            str(output_path),
        ],
    )
    build_dataset_card.main()

    assert output_path.exists()
    assert "Test Mixture" in output_path.read_text(encoding="utf-8")
