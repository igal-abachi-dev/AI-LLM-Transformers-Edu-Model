import json
import random

import numpy as np
import pytest
import torch

from minifrontier.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from minifrontier.run_metadata import RunMetadata


def test_seed_everything_repeats_all_cpu_generators() -> None:
    seed_everything(42)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(42)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_rng_state_round_trip() -> None:
    seed_everything(7)
    state = capture_rng_state()
    expected = torch.rand(4)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), expected)


def test_run_metadata_is_validated_and_json_safe(tmp_path) -> None:
    record = RunMetadata(name="test", config={"width": 16}, seed=42, parameters=123)
    path = tmp_path / "run.json"
    record.write_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "test"
    assert data["parameters"] == 123
    assert data["torch_version"] == torch.__version__


def test_run_metadata_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RunMetadata(name="bad", config={}, seed=42, parameters=-1)
