import json

import pytest
import torch

from minifrontier.checkpoint import (
    export_release,
    load_release,
    load_training_checkpoint,
    prune_old_checkpoints,
    save_training_checkpoint,
)
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier


def train_step(model, optimizer, scheduler, tokens) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = model(tokens, labels=tokens).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def test_checkpoint_exact_resume_and_trust_boundary(tmp_path) -> None:
    torch.manual_seed(15)
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    train_step(model, optimizer, scheduler, tokens)
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"step": 1},
        data_cursor={"document": 7},
    )
    expected_random = torch.rand(4)
    expected_loss = train_step(model, optimizer, scheduler, tokens)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    resumed = MiniFrontier(config)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda step: 1.0)
    try:
        load_training_checkpoint(checkpoint, resumed, optimizer=resumed_optimizer)
    except ValueError as error:
        assert "pickle" in str(error)
    trainer_state, cursor = load_training_checkpoint(
        checkpoint,
        resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        trusted_local_state=True,
    )
    actual_random = torch.rand(4)
    actual_loss = train_step(resumed, resumed_optimizer, resumed_scheduler, tokens)
    assert torch.equal(actual_random, expected_random)
    assert actual_loss == expected_loss
    assert trainer_state == {"step": 1}
    assert cursor == {"document": 7}
    for name, value in resumed.state_dict().items():
        assert torch.equal(value, expected[name])
    assert resumed.lm_head.weight is resumed.token_embedding.weight


def test_interrupted_checkpoint_save_never_corrupts_target(tmp_path, monkeypatch) -> None:
    """A save killed partway through must never leave a partial checkpoint at `directory`."""

    import minifrontier.checkpoint as checkpoint_module

    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    checkpoint = tmp_path / "checkpoint"
    real_torch_save = checkpoint_module.torch.save

    # Case 1: interrupting the very first save at this path must leave nothing
    # behind that looks like a checkpoint (no directory, or only the hidden
    # staging one) -- never a `directory` with some files but not others.
    def broken_torch_save(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated kill mid-write")

    monkeypatch.setattr(checkpoint_module.torch, "save", broken_torch_save)
    with pytest.raises(RuntimeError, match="simulated kill mid-write"):
        save_training_checkpoint(checkpoint, model, trainer_state={"step": 0})
    assert not checkpoint.exists()

    # Case 2: a real, complete checkpoint already exists; interrupting an
    # OVERWRITE of it must leave the original, still fully loadable.
    monkeypatch.setattr(checkpoint_module.torch, "save", real_torch_save)
    save_training_checkpoint(checkpoint, model, trainer_state={"step": 1})
    original_files = sorted(path.name for path in checkpoint.iterdir())

    monkeypatch.setattr(checkpoint_module.torch, "save", broken_torch_save)
    with pytest.raises(RuntimeError, match="simulated kill mid-write"):
        save_training_checkpoint(checkpoint, model, trainer_state={"step": 2})
    assert sorted(path.name for path in checkpoint.iterdir()) == original_files
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert trainer_state == {"step": 1}

    # A stale staging directory from the failed attempt may still be on disk
    # (cleaned up lazily by the next save, not immediately) -- but a THIRD,
    # successful save must still work correctly despite it being there.
    monkeypatch.setattr(checkpoint_module.torch, "save", real_torch_save)
    save_training_checkpoint(checkpoint, model, trainer_state={"step": 3})
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert trainer_state == {"step": 3}
    assert not any(path.name.startswith(".checkpoint") for path in tmp_path.iterdir())


def test_prune_old_checkpoints_keeps_only_most_recent_and_never_touches_final(tmp_path) -> None:
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    output = tmp_path / "run"
    for updates in (100, 200, 300, 400, 500):
        save_training_checkpoint(
            output / f"checkpoint-{updates:08d}", model, trainer_state={"step": updates}
        )
    save_training_checkpoint(output / "final", model, trainer_state={"step": 500})
    # An unrelated directory that happens to share a prefix must be left alone.
    (output / "checkpoint-notanumber").mkdir()

    deleted = prune_old_checkpoints(output, keep_last_n=2)

    assert {path.name for path in deleted} == {
        "checkpoint-00000100",
        "checkpoint-00000200",
        "checkpoint-00000300",
    }
    remaining = {path.name for path in output.iterdir()}
    assert remaining == {
        "checkpoint-00000400",
        "checkpoint-00000500",
        "final",
        "checkpoint-notanumber",
    }

    # Idempotent: pruning again with nothing left to delete is a no-op, not an error.
    assert prune_old_checkpoints(output, keep_last_n=2) == []

    with pytest.raises(ValueError, match="positive"):
        prune_old_checkpoints(output, keep_last_n=0)


def test_release_folder_loads_independently(tmp_path, mini_tokenizer) -> None:
    torch.manual_seed(16)
    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    model = MiniFrontier(config).eval()
    tokens = torch.tensor([[1, 20, 30]])
    expected = model(tokens).logits
    release = tmp_path / "release"
    export_release(release, model, mini_tokenizer)
    loaded, loaded_tokenizer = load_release(release)
    assert torch.equal(loaded(tokens).logits, expected)
    assert loaded.lm_head.weight is loaded.token_embedding.weight
    assert loaded_tokenizer.encode("hello") == mini_tokenizer.encode("hello")
    tokenizer_config = json.loads((release / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["model_max_length"] == config.max_seq_len
    assert {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
        "chat_template.jinja",
        "system_prompt.md",
        "generation_config.json",
        "sha256-manifest.json",
    } <= {path.name for path in release.iterdir()}


def test_release_manifest_rejects_tampering(tmp_path, mini_tokenizer) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    release = tmp_path / "release"
    export_release(release, MiniFrontier(config), mini_tokenizer)
    (release / "generation_config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_release(release)


def test_training_checkpoint_rejects_same_shape_semantic_config_change(tmp_path) -> None:
    config = ModelConfig.tiny_modern(attention_impl="sdpa")
    model = MiniFrontier(config)
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, model)
    incompatible = MiniFrontier(
        ModelConfig.tiny_modern(
            attention_impl="sdpa",
            global_position_encoding="none",
        )
    )
    with pytest.raises(ValueError, match="configuration"):
        load_training_checkpoint(checkpoint, incompatible)
