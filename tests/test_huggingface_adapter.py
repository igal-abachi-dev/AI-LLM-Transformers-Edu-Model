from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    dynamic_module_utils,
)

from minifrontier.adapters.huggingface.modeling_minifrontier import MiniFrontierForCausalLM
from minifrontier.checkpoint import export_release
from minifrontier.config import ModelConfig
from minifrontier.hf_export import (
    compare_native_transformers,
    export_transformers_repository,
    load_native_weights_into_transformers,
    transformers_config,
)
from minifrontier.model import MiniFrontier


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig.tiny_edu(attention_impl="manual"),
        ModelConfig.tiny_modern(attention_impl="manual"),
        replace(
            ModelConfig.tiny_modern(attention_impl="manual"),
            global_position_encoding="none",
        ),
        # MF-081: split local/global RoPE theta must reach the HF export path
        # the same way head_dim_override did (see the case below) -- exercised
        # with a theta divergent enough that a missed wiring bug would produce
        # very different (not just numerically-close) logits.
        replace(
            ModelConfig.tiny_modern(attention_impl="manual"),
            global_rope_theta=1_000_000.0,
        ),
        # MF-081: LayerNorm scaling and value residual must also reach the HF
        # export path with real end-to-end parity, not just field mirroring.
        replace(ModelConfig.tiny_modern(attention_impl="manual"), layer_norm_scaling=True),
        replace(ModelConfig.tiny_modern(attention_impl="manual"), value_residual=True),
        # MF-082: per-head gated attention must also reach the HF export path.
        replace(ModelConfig.tiny_modern(attention_impl="manual"), gated_attention=True),
        # MF-106: SwiGLU clamping, on Edu -- unlike the items above, this one
        # is NOT Modern-only, so this is also real coverage that the Edu
        # guard change didn't accidentally over-restrict it. A small clamp
        # value so it actually binds against real activations, not just
        # exists as an inert field.
        replace(ModelConfig.tiny_edu(attention_impl="manual"), swiglu_clamp=0.05),
        # d_model=48 is not divisible by n_heads=5 -- only valid because of
        # head_dim_override (MF-092). This is the regression case for the real
        # gap found by a third code-review round: the HF adapter used to
        # hardcode head_dim = d_model // n_heads and discard the override
        # entirely, either raising on this exact divisibility or silently
        # producing wrong-shaped projections.
        ModelConfig.tiny_edu(d_model=48, n_heads=5, head_dim_override=16, attention_impl="manual"),
    ],
)
def test_native_transformers_logits_and_greedy_tokens_match(config, mini_tokenizer) -> None:
    config = replace(config, vocab_size=max(512, mini_tokenizer.vocab_size))
    torch.manual_seed(9)
    native = MiniFrontier(config).eval()
    if config.value_residual:
        # A zero gate is a no-op regardless of whether the mixing arithmetic
        # is wired correctly -- perturb it so a real wiring bug would produce
        # a genuine logit mismatch, not a false-positive pass.
        with torch.no_grad():
            for block in native.blocks[1:]:
                block.attention.value_residual_gate.fill_(0.5)
    if config.gated_attention:
        # At init gate_proj.weight is zero, making the gate a spatially
        # UNIFORM constant -- a wrong axis alignment (e.g. swapped head/
        # sequence axes) would be invisible against a uniform multiplier.
        # Perturb the weight so the gate is genuinely input- and head-
        # dependent, the scenario that would actually expose such a bug.
        torch.manual_seed(17)
        with torch.no_grad():
            for block in native.blocks:
                block.attention.gate_proj.weight.normal_(mean=0.0, std=0.5)
    hf_config = transformers_config(native, mini_tokenizer)
    hf_model = MiniFrontierForCausalLM(hf_config).eval()
    load_native_weights_into_transformers(native, hf_model)
    tokens = torch.randint(0, config.vocab_size, (2, 11))
    parity = compare_native_transformers(native, hf_model, tokens)
    assert parity["allclose"], parity
    assert parity["argmax_equal"], parity


# attention_impl is deliberately dropped by transformers_config (it is a
# native-only runtime-kernel choice, not an architecture field -- the HF
# config always sets its own _attn_implementation="eager" separately). Every
# other ModelConfig field must survive onto MiniFrontierConfig under the same
# attribute name, or a future field addition silently fails to reach the HF
# export path the way head_dim_override did (found by a third code-review
# round: MiniFrontierConfig hardcoded head_dim = d_model // n_heads and
# discarded the override entirely).
_FIELDS_NOT_MIRRORED_ONTO_HF_CONFIG = frozenset({"attention_impl"})


def test_transformers_config_mirrors_every_modelconfig_field(mini_tokenizer) -> None:
    config = ModelConfig.tiny_edu(
        vocab_size=max(512, mini_tokenizer.vocab_size),
        d_model=48,
        n_heads=5,
        head_dim_override=16,
        attention_impl="manual",
    )
    native = MiniFrontier(config)
    hf_config = transformers_config(native, mini_tokenizer)
    for field in fields(ModelConfig):
        if field.name in _FIELDS_NOT_MIRRORED_ONTO_HF_CONFIG:
            continue
        assert hasattr(hf_config, field.name), f"{field.name} missing from MiniFrontierConfig"
        expected = getattr(config, field.name)
        actual = getattr(hf_config, field.name)
        assert actual == expected, f"{field.name}: ModelConfig={expected!r}, HF config={actual!r}"


def test_transformers_cached_generation_matches_full_forward(mini_tokenizer) -> None:
    config = ModelConfig.tiny_modern(
        vocab_size=max(512, mini_tokenizer.vocab_size), attention_impl="manual"
    )
    native = MiniFrontier(config)
    model = MiniFrontierForCausalLM(transformers_config(native, mini_tokenizer)).eval()
    load_native_weights_into_transformers(native, model)
    tokens = torch.randint(0, config.vocab_size, (1, 10))
    expected = model(input_ids=tokens, use_cache=False).logits
    cached = model(input_ids=tokens[:, :6], use_cache=True)
    actual_tail = model(
        input_ids=tokens[:, 6:],
        past_key_values=cached.past_key_values,
        use_cache=True,
    ).logits
    assert torch.allclose(expected[:, 6:], actual_tail, atol=2e-5, rtol=0)
    assert torch.equal(expected[:, 6:].argmax(-1), actual_tail.argmax(-1))


def test_transformers_model_exposes_vllm_backend_contract(mini_tokenizer) -> None:
    native = MiniFrontier(
        ModelConfig.tiny_modern(
            vocab_size=max(512, mini_tokenizer.vocab_size), attention_impl="sdpa"
        )
    )
    config = transformers_config(native, mini_tokenizer)
    model = MiniFrontierForCausalLM(config)
    assert model._supports_attention_backend is True
    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert config.auto_map["AutoModel"] == "modeling_minifrontier.MiniFrontierModel"


def test_hub_repository_auto_classes_load_locally(
    tmp_path: Path, mini_tokenizer, monkeypatch
) -> None:
    native_release = tmp_path / "native"
    output = tmp_path / "hub"
    config = ModelConfig.tiny_modern(
        vocab_size=max(512, mini_tokenizer.vocab_size), attention_impl="sdpa"
    )
    export_release(native_release, MiniFrontier(config), mini_tokenizer)
    report = export_transformers_repository(
        native_release,
        output,
        source_revision="0123456789abcdef0123456789abcdef01234567",
    )
    assert report["local_auto_load"] == "passed"
    monkeypatch.setattr(
        dynamic_module_utils,
        "HF_MODULES_CACHE",
        str(tmp_path / "hf-modules-cache"),
    )
    loaded_config = AutoConfig.from_pretrained(output, trust_remote_code=True)
    assert loaded_config.__class__.__name__ == "MiniFrontierConfig"
    assert loaded_config.model_type == "minifrontier"
    assert AutoModel.from_pretrained(output, trust_remote_code=True).__class__.__name__ == (
        "MiniFrontierModel"
    )
    assert (
        AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True).__class__.__name__
        == "MiniFrontierForCausalLM"
    )
    tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    assert tokenizer.bos_token_id == mini_tokenizer.bos_id
    assert tokenizer.eos_token_id == mini_tokenizer.eos_id
    assert (output / "sha256-manifest.json").exists()
