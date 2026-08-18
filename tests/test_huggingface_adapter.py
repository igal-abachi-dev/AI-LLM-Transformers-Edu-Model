from __future__ import annotations

from dataclasses import replace
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
    ],
)
def test_native_transformers_logits_and_greedy_tokens_match(config, mini_tokenizer) -> None:
    config = replace(config, vocab_size=max(512, mini_tokenizer.vocab_size))
    torch.manual_seed(9)
    native = MiniFrontier(config).eval()
    hf_config = transformers_config(native, mini_tokenizer)
    hf_model = MiniFrontierForCausalLM(hf_config).eval()
    load_native_weights_into_transformers(native, hf_model)
    tokens = torch.randint(0, config.vocab_size, (2, 11))
    parity = compare_native_transformers(native, hf_model, tokens)
    assert parity["allclose"], parity
    assert parity["argmax_equal"], parity


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
