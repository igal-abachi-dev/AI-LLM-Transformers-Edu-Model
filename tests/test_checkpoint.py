import json

import pytest
import torch

from minifrontier.checkpoint import (
    export_release,
    load_release,
    load_release_mtp_heads,
    load_training_checkpoint,
    prune_old_checkpoints,
    save_training_checkpoint,
)
from minifrontier.config import ModelConfig
from minifrontier.ema import EMAWeights
from minifrontier.model import MiniFrontier
from minifrontier.mtp import MTPHeads


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


def test_mtp_heads_round_trip_through_checkpoint_save_and_load(tmp_path) -> None:
    torch.manual_seed(21)
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    mtp_heads = MTPHeads(d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=1)
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, model, mtp_heads=mtp_heads)
    assert (checkpoint / "mtp_heads.safetensors").exists()

    expected = {name: value.detach().clone() for name, value in mtp_heads.state_dict().items()}
    loaded_model = MiniFrontier(config)
    loaded_heads = MTPHeads(d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=1)
    load_training_checkpoint(
        checkpoint, loaded_model, mtp_heads=loaded_heads, trusted_local_state=True
    )
    for name, value in loaded_heads.state_dict().items():
        assert torch.equal(value, expected[name])


def test_load_training_checkpoint_rejects_missing_mtp_heads_instead_of_silent_reinit(
    tmp_path,
) -> None:
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, model)  # no mtp_heads=... passed
    requesting_heads = MTPHeads(
        d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=1
    )
    with pytest.raises(ValueError, match=r"mtp_heads\.safetensors"):
        load_training_checkpoint(checkpoint, MiniFrontier(config), mtp_heads=requesting_heads)


def test_ema_round_trips_through_checkpoint_save_and_load(tmp_path) -> None:
    torch.manual_seed(22)
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    ema = EMAWeights(model, decay=0.9)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, model, ema=ema)
    assert (checkpoint / "ema.safetensors").exists()

    expected = dict(ema.state_dict())
    loaded_model = MiniFrontier(config)
    loaded_ema = EMAWeights(loaded_model, decay=0.9)
    load_training_checkpoint(checkpoint, loaded_model, ema=loaded_ema, trusted_local_state=True)
    for name, value in loaded_ema.state_dict().items():
        assert torch.equal(value, expected[name])


def test_load_training_checkpoint_rejects_missing_ema_instead_of_silent_reinit(tmp_path) -> None:
    config = ModelConfig.tiny_edu()
    model = MiniFrontier(config)
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, model)  # no ema=... passed
    requesting_ema = EMAWeights(MiniFrontier(config), decay=0.9)
    with pytest.raises(ValueError, match=r"ema\.safetensors"):
        load_training_checkpoint(checkpoint, MiniFrontier(config), ema=requesting_ema)


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


def test_release_without_mtp_heads_has_none_and_no_mtp_files(tmp_path, mini_tokenizer) -> None:
    config = ModelConfig.tiny_edu(vocab_size=max(512, mini_tokenizer.vocab_size))
    release = tmp_path / "release"
    export_release(release, MiniFrontier(config), mini_tokenizer)
    assert load_release_mtp_heads(release, config) is None
    assert not (release / "mtp_heads.safetensors").exists()
    assert not (release / "mtp_config.json").exists()


def test_release_with_mtp_heads_round_trips_real_weights(tmp_path, mini_tokenizer) -> None:
    torch.manual_seed(3)
    config = ModelConfig.tiny_modern(vocab_size=max(512, mini_tokenizer.vocab_size))
    mtp_heads = MTPHeads(d_model=config.d_model, vocab_size=config.vocab_size, n_extra_heads=1)
    release = tmp_path / "release"
    export_release(release, MiniFrontier(config), mini_tokenizer, mtp_heads=mtp_heads)
    assert (release / "mtp_heads.safetensors").exists()
    assert (release / "mtp_config.json").exists()

    loaded_heads = load_release_mtp_heads(release, config)
    assert loaded_heads is not None
    assert loaded_heads.n_extra_heads == 1
    hidden = torch.randn(1, 3, config.d_model)
    assert torch.equal(loaded_heads.predict(hidden), mtp_heads.predict(hidden))

    # sha256-manifest.json covers the two new files too, not just the original set.
    manifest = json.loads((release / "sha256-manifest.json").read_text(encoding="utf-8"))
    assert "mtp_heads.safetensors" in manifest
    assert "mtp_config.json" in manifest


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


def test_training_checkpoint_loads_after_a_new_modelconfig_field_is_added(tmp_path) -> None:
    """A checkpoint's config.json predates a later additive ModelConfig field
    (e.g. rope_fraction, added by MF-107 after several real MF-081 arms were
    already trained). Simulate that by hand-writing a config.json with the
    field missing, matching what an old checkpoint on disk actually looks
    like, and confirm it still loads under the current schema."""

    config = ModelConfig.tiny_modern(attention_impl="sdpa")
    model = MiniFrontier(config)
    checkpoint = tmp_path / "checkpoint"
    save_training_checkpoint(checkpoint, model)
    saved = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert "rope_fraction" in saved
    del saved["rope_fraction"]
    (checkpoint / "config.json").write_text(json.dumps(saved), encoding="utf-8")

    load_training_checkpoint(checkpoint, MiniFrontier(config), trusted_local_state=True)
