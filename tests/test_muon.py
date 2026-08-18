import torch

from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.muon import (
    build_muon_adamw,
    newton_schulz_reference,
    partition_muon_parameters,
)
from minifrontier.training import (
    ListBatchProvider,
    TrainingBatch,
    TrainingConfig,
    train_updates,
)


def test_newton_schulz_reference_is_finite_shape_preserving_and_non_mutating() -> None:
    torch.manual_seed(55)
    matrix = torch.randn(4, 7)
    original = matrix.clone()
    result = newton_schulz_reference(matrix, steps=5)
    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()
    assert torch.equal(matrix, original)
    singular_values = torch.linalg.svdvals(result.float())
    assert torch.all((singular_values > 0.4) & (singular_values < 1.4))


def test_muon_partition_is_disjoint_complete_and_excludes_embedding() -> None:
    model = MiniFrontier(ModelConfig.tiny_edu())
    partition = partition_muon_parameters(model)
    muon_ids = {id(parameter) for parameter in partition.muon_parameters}
    adamw_ids = {id(parameter) for parameter in partition.adamw_parameters}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert not muon_ids & adamw_ids
    assert muon_ids | adamw_ids == expected
    assert "token_embedding.weight" in partition.adamw_names
    assert all(name != "token_embedding.weight" for name in partition.muon_names)
    assert all(parameter.ndim == 2 for parameter in partition.muon_parameters)


def test_first_party_muon_and_adamw_train_and_checkpoint_state() -> None:
    torch.manual_seed(56)
    config = ModelConfig.tiny_edu(n_layers=1, d_model=16, n_heads=2, d_ff=32)
    model = MiniFrontier(config)
    training = TrainingConfig(
        max_updates=2,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_updates=0,
    )
    optimizer, report = build_muon_adamw(
        model,
        training,
        muon_learning_rate=1e-3,
        adamw_learning_rate=3e-4,
        match_rms_adamw=True,
    )
    provider = ListBatchProvider([TrainingBatch(torch.randint(0, config.vocab_size, (2, 8)))])
    optimizer, _, state, _ = train_updates(
        model,
        provider,
        training,
        optimizer=optimizer,
    )
    assert state.completed_updates == 2
    assert report["implementation"] == "torch.optim.Muon"
    assert report["muon_parameters"] > report["adamw_parameters"]
    assert report["adamw_decay_names"] == ["token_embedding.weight"]
    adamw_groups = optimizer.optimizers[1].param_groups
    assert [group["weight_decay"] for group in adamw_groups] == [
        training.weight_decay,
        0.0,
    ]
    saved = optimizer.state_dict()
    replacement, _ = build_muon_adamw(
        model,
        training,
        muon_learning_rate=1e-3,
        adamw_learning_rate=3e-4,
    )
    replacement.load_state_dict(saved)
