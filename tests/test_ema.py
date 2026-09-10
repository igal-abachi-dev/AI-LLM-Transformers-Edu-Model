import pytest
import torch

from minifrontier.config import ModelConfig
from minifrontier.ema import EMAWeights
from minifrontier.model import MiniFrontier


def test_decay_must_be_in_open_unit_interval() -> None:
    model = MiniFrontier(ModelConfig.tiny_edu())
    for bad_decay in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="decay"):
            EMAWeights(model, decay=bad_decay)


def test_initial_shadow_matches_but_does_not_alias_live_parameters() -> None:
    torch.manual_seed(1)
    model = MiniFrontier(ModelConfig.tiny_edu())
    ema = EMAWeights(model, decay=0.9)
    for name, parameter in model.named_parameters():
        assert torch.equal(ema.state_dict()[name], parameter.detach())
    # Mutating the live model must not move the shadow -- it was cloned, not aliased.
    first_name = next(name for name, _ in model.named_parameters())
    with torch.no_grad():
        dict(model.named_parameters())[first_name].add_(1.0)
    assert not torch.equal(ema.state_dict()[first_name], dict(model.named_parameters())[first_name])


def test_tied_parameters_are_deduplicated_in_the_shadow() -> None:
    config = ModelConfig.tiny_edu()
    assert config.tie_embeddings
    model = MiniFrontier(config)
    ema = EMAWeights(model, decay=0.9)
    shadow_names = set(ema.state_dict())
    # model.state_dict() (unlike named_parameters(), which already deduplicates
    # tied tensors by identity) lists both tied names -- the shadow must not.
    module_state_names = set(model.state_dict())
    assert "token_embedding.weight" in module_state_names
    assert "lm_head.weight" in module_state_names
    assert "token_embedding.weight" in shadow_names
    assert "lm_head.weight" not in shadow_names
    assert len(shadow_names) < len(module_state_names)


def test_update_moves_shadow_the_documented_fraction_toward_live_weights() -> None:
    torch.manual_seed(2)
    model = MiniFrontier(ModelConfig.tiny_edu())
    ema = EMAWeights(model, decay=0.9)
    name = next(name for name, _ in model.named_parameters())
    before = ema.state_dict()[name].clone()
    with torch.no_grad():
        dict(model.named_parameters())[name].add_(1.0)
    ema.update(model)
    live = dict(model.named_parameters())[name]
    expected = before * 0.9 + live * 0.1
    assert torch.allclose(ema.state_dict()[name], expected)


def test_copy_to_overwrites_live_model_with_shadow_weights() -> None:
    torch.manual_seed(3)
    model = MiniFrontier(ModelConfig.tiny_edu())
    ema = EMAWeights(model, decay=0.9)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.copy_to(model)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), ema.state_dict()[name])


def test_save_and_load_round_trips_shadow_weights(tmp_path) -> None:
    torch.manual_seed(4)
    model = MiniFrontier(ModelConfig.tiny_edu())
    ema = EMAWeights(model, decay=0.9)
    ema.update(model)
    path = str(tmp_path / "ema.safetensors")
    ema.save(path)

    reloaded = EMAWeights(MiniFrontier(ModelConfig.tiny_edu()), decay=0.9)
    reloaded.load(path)
    for name, tensor in ema.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name], tensor)


def test_load_rejects_a_shadow_with_the_wrong_parameter_names(tmp_path) -> None:
    torch.manual_seed(5)
    model = MiniFrontier(ModelConfig.tiny_edu())
    ema = EMAWeights(model, decay=0.9)
    path = str(tmp_path / "ema.safetensors")
    ema.save(path)

    other_config = ModelConfig.tiny_edu(n_layers=ModelConfig.tiny_edu().n_layers + 1)
    mismatched = EMAWeights(MiniFrontier(other_config), decay=0.9)
    with pytest.raises(ValueError, match="parameter names"):
        mismatched.load(path)
