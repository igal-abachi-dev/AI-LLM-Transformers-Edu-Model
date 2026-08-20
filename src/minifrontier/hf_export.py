"""Hub-ready Transformers export without changing the native MiniFrontier checkpoint.

Beginner's map of this file
---------------------------
Writes a second copy of a release in the layout the Transformers library expects,
so ``AutoModelForCausalLM.from_pretrained(...)`` works on it. The raw-PyTorch
checkpoint stays the reference implementation and is not modified -- this is a
translation for other people's tools, not a dependency of the model itself.

The export has to earn its claim through parity tests: same prompt, same outputs
as the native path, or the export is wrong.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerFast,
    dynamic_module_utils,
)

from minifrontier.adapters.huggingface.configuration_minifrontier import MiniFrontierConfig
from minifrontier.adapters.huggingface.modeling_minifrontier import MiniFrontierForCausalLM
from minifrontier.checkpoint import load_release, verify_release_manifest, write_release_manifest
from minifrontier.model import MiniFrontier
from minifrontier.tokenizer import SPECIAL_TOKENS, MiniFrontierTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transformers_config(
    model: MiniFrontier,
    tokenizer: MiniFrontierTokenizer,
) -> MiniFrontierConfig:
    values = model.config.to_dict()
    values.pop("attention_impl")
    config = MiniFrontierConfig(
        **values,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )
    config.architectures = ["MiniFrontierForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_minifrontier.MiniFrontierConfig",
        "AutoModel": "modeling_minifrontier.MiniFrontierModel",
        "AutoModelForCausalLM": "modeling_minifrontier.MiniFrontierForCausalLM",
    }
    config._attn_implementation = "eager"
    return config


def load_native_weights_into_transformers(
    native: MiniFrontier,
    target: MiniFrontierForCausalLM,
) -> None:
    mapped = {
        (key if key == "lm_head.weight" else f"model.{key}"): value
        for key, value in native.state_dict().items()
    }
    missing, unexpected = target.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"native/Transformers state mapping failed; missing={missing}, unexpected={unexpected}"
        )
    if native.config.tie_embeddings:
        target.tie_weights()


def compare_native_transformers(
    native: MiniFrontier,
    transformers_model: MiniFrontierForCausalLM,
    tokens: torch.Tensor,
    *,
    atol: float = 2e-5,
) -> dict[str, object]:
    native.eval()
    transformers_model.eval()
    with torch.no_grad():
        expected = native(tokens, attention_impl="manual").logits
        actual = transformers_model(input_ids=tokens, use_cache=False).logits
    difference = (expected - actual).abs().max().item()
    return {
        "max_absolute_logit_difference": difference,
        "allclose": bool(torch.allclose(expected, actual, atol=atol, rtol=0)),
        "argmax_equal": bool(torch.equal(expected.argmax(-1), actual.argmax(-1))),
    }


def _write_hf_tokenizer(
    source: Path,
    target: Path,
    tokenizer: MiniFrontierTokenizer,
    *,
    model_max_length: int,
) -> None:
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(source / "tokenizer.json"),
        bos_token=SPECIAL_TOKENS[tokenizer.bos_id],
        eos_token=SPECIAL_TOKENS[tokenizer.eos_id],
        pad_token=SPECIAL_TOKENS[tokenizer.pad_id],
        additional_special_tokens=list(SPECIAL_TOKENS[3:]),
        model_max_length=model_max_length,
    )
    template_path = source / "chat_template.jinja"
    if template_path.exists():
        fast.chat_template = template_path.read_text(encoding="utf-8")
    fast.save_pretrained(target)


def export_transformers_repository(
    native_release: str | Path,
    output: str | Path,
    *,
    source_revision: str,
    model_card: str | None = None,
) -> dict[str, Any]:
    """Export a standalone remote-code repository and verify local Auto loading/parity."""

    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("source_revision must be a full lowercase 40-character Git commit")
    source = Path(native_release)
    target = Path(output)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Transformers export target must be absent or empty")
    verify_release_manifest(source)
    native, tokenizer = load_release(source)
    target.mkdir(parents=True, exist_ok=True)
    config = transformers_config(native, tokenizer)
    hf_model = MiniFrontierForCausalLM(config)
    load_native_weights_into_transformers(native, hf_model)
    hf_model.save_pretrained(target, safe_serialization=True)
    _write_hf_tokenizer(
        source,
        target,
        tokenizer,
        model_max_length=native.config.max_seq_len,
    )
    GenerationConfig(
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
        max_length=native.config.max_seq_len,
        do_sample=False,
    ).save_pretrained(target)
    adapter_source = Path(__file__).parent / "adapters" / "huggingface"
    for name in ("configuration_minifrontier.py", "modeling_minifrontier.py"):
        shutil.copyfile(adapter_source / name, target / name)
    for name in ("chat_template.jinja", "system_prompt.md"):
        path = source / name
        if path.exists():
            shutil.copyfile(path, target / name)
    if (source / "LICENSE").exists():
        shutil.copyfile(source / "LICENSE", target / "LICENSE")
    card = model_card or (
        "# MiniFrontier Transformers export\n\n"
        "Standalone remote-code export of a native MiniFrontier release. Use a pinned revision and "
        "review the Python modeling files before enabling `trust_remote_code=True`. This text-only "
        "educational model does not claim tool calling, production safety, or frontier quality.\n"
    )
    (target / "README.md").write_text(card, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "format": "minifrontier-transformers",
        "source_revision": source_revision,
        "source_manifest_sha256": _sha256(source / "sha256-manifest.json"),
        "transformers_attention_interface": True,
        "trust_remote_code_required": True,
        "vllm_validation": "unmeasured",
        "native_win32_vllm_supported": False,
    }
    (target / "minifrontier_adapter.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_release_manifest(target)

    original_modules_cache = dynamic_module_utils.HF_MODULES_CACHE
    with tempfile.TemporaryDirectory(prefix="minifrontier-hf-modules-") as modules_cache:
        dynamic_module_utils.HF_MODULES_CACHE = modules_cache
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
            loaded_model = AutoModelForCausalLM.from_pretrained(target, trust_remote_code=True)
            prompt = torch.tensor([[tokenizer.bos_id, tokenizer.eos_id]], dtype=torch.long)
            parity = compare_native_transformers(native, loaded_model, prompt)
            if not parity["allclose"] or not parity["argmax_equal"]:
                raise RuntimeError(f"native/Transformers local parity failed: {parity}")
            if loaded_tokenizer.get_vocab() != tokenizer.backend.get_vocab():
                raise RuntimeError("native/Transformers tokenizer vocabularies differ")
        finally:
            dynamic_module_utils.HF_MODULES_CACHE = original_modules_cache
    return {**metadata, "local_auto_load": "passed", "parity": parity}
