"""Schemas and subprocess/API boundaries for post-V1 external-runtime validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExternalValidationRecord:
    runtime: Literal["vllm", "llama.cpp", "gguf-quantize"]
    status: Literal["passed", "failed", "unmeasured"]
    runtime_revision: str
    model_revision: str
    command: tuple[str, ...]
    hardware: str
    precision: str
    parity: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.runtime_revision or not self.model_revision:
            raise ValueError("external validation requires pinned runtime and model revisions")
        if not self.command:
            raise ValueError("external validation requires the exact command")
        if self.status == "passed":
            if self.error is not None:
                raise ValueError("a passed record cannot contain an error")
            if not self.parity or not self.metrics or not self.artifacts:
                raise ValueError("passed external validation requires parity, metrics, and hashes")
            serialized = json.dumps(
                {"parity": self.parity, "metrics": self.metrics}, sort_keys=True
            )
            if "unmeasured" in serialized:
                raise ValueError("passed external validation cannot contain unmeasured fields")

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def run_checked(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], float]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=Path(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return completed, time.perf_counter() - started


def openai_request(
    base_url: str,
    endpoint: str,
    payload: Mapping[str, Any] | None,
    *,
    api_key: str,
    timeout_seconds: float = 120.0,
) -> tuple[dict[str, Any], float]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    body = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="GET" if payload is None else "POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI-compatible endpoint returned HTTP {error.code}: {detail}"
        ) from error
    elapsed = time.perf_counter() - started
    if not isinstance(value, dict):
        raise ValueError("OpenAI-compatible response must be a JSON object")
    return value, elapsed


def validate_vllm_api(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    system_prompt: str,
    parity_fixture: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    models, models_seconds = openai_request(base_url, "/models", None, api_key=api_key)
    completion, completion_seconds = openai_request(
        base_url,
        "/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": int(parity_fixture.get("max_new_tokens", 8)) if parity_fixture else 8,
            "temperature": 0,
            "logprobs": 1,
            "prompt_logprobs": 1,
        },
        api_key=api_key,
    )
    chat, chat_seconds = openai_request(
        base_url,
        "/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 8,
            "temperature": 0,
        },
        api_key=api_key,
    )
    if not completion.get("choices") or not chat.get("choices"):
        raise ValueError("vLLM responses must contain non-empty choices")
    served_ids = {item.get("id") for item in models.get("data", [])}
    if model not in served_ids:
        raise ValueError(f"requested model {model!r} is absent from /models")
    parity: dict[str, object] = {"status": "not_requested"}
    if parity_fixture is not None:
        choice = completion["choices"][0]
        actual_text = choice.get("text")
        expected_text = parity_fixture.get("expected_completion_text")
        prompt_logprobs_present = bool(
            choice.get("prompt_logprobs") or completion.get("prompt_logprobs")
        )
        parity = {
            "status": "passed"
            if actual_text == expected_text and prompt_logprobs_present
            else "failed",
            "greedy_text_equal": actual_text == expected_text,
            "prompt_logprobs_present": prompt_logprobs_present,
            "expected_completion_text": expected_text,
            "actual_completion_text": actual_text,
        }
    return {
        "transport_status": "passed",
        "model_list_seconds": models_seconds,
        "completion_seconds": completion_seconds,
        "chat_seconds": chat_seconds,
        "completion": completion,
        "chat": chat,
        "native_parity": parity,
        "tool_calling_claim": False,
        "code_edit_quality_claim": False,
    }
