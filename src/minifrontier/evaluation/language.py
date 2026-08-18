"""lm-evaluation-harness adapter for MiniFrontier (MF-035)."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import torch

try:
    from lm_eval.api.model import LM
except ImportError:  # pragma: no cover - exercised in the core-only installation

    class LM:  # type: ignore[no-redef]
        """Minimal base so validation and adapter smoke remain usable without lm-eval."""

        def __init__(self) -> None:
            self._device: torch.device | None = None

        @property
        def device(self) -> torch.device | None:
            return self._device


from minifrontier.model import MiniFrontier
from minifrontier.tokenizer import MiniFrontierTokenizer

DEFAULT_TASKS = ("arc_easy", "hellaswag", "piqa")
OPTIONAL_TASKS = ("gsm8k",)


class MiniFrontierEvalLM(LM):
    """Correctness-first single-device adapter for the standard harness API."""

    def __init__(
        self,
        model: MiniFrontier,
        tokenizer: MiniFrontierTokenizer,
        *,
        max_gen_tokens: int = 64,
    ) -> None:
        super().__init__()
        if max_gen_tokens <= 0:
            raise ValueError("max_gen_tokens must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_gen_tokens = max_gen_tokens
        self._device = next(model.parameters()).device

    @property
    def tokenizer_name(self) -> str:
        return "minifrontier-byte-bpe-v1"

    @property
    def max_length(self) -> int:
        return self.model.config.max_seq_len

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_id

    @torch.inference_mode()
    def _score(self, prefix: list[int], continuation: list[int]) -> tuple[float, bool]:
        history = [self.tokenizer.bos_id, *prefix]
        log_probability = 0.0
        is_greedy = True
        for target in continuation:
            context = history[-self.max_length :]
            tokens = torch.tensor([context], dtype=torch.long, device=self.device)
            next_logits = self.model(tokens).logits[0, -1].float()
            log_probability += float(torch.log_softmax(next_logits, dim=-1)[target].cpu())
            is_greedy = is_greedy and int(next_logits.argmax().cpu()) == target
            history.append(target)
        return log_probability, is_greedy

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        results = []
        for request in requests:
            context, continuation = request.args
            results.append(
                self._score(
                    self.tokenizer.encode(context),
                    self.tokenizer.encode(continuation),
                )
            )
        return results

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        return [self._score([], self.tokenizer.encode(request.args[0]))[0] for request in requests]

    @torch.inference_mode()
    def generate_until(self, requests: list[Any]) -> list[str]:
        results = []
        for request in requests:
            context, kwargs = request.args
            until = kwargs.get("until", [])
            stop_strings = [until] if isinstance(until, str) else list(until)
            requested = int(kwargs.get("max_gen_toks", self.max_gen_tokens))
            max_new_tokens = min(requested, self.max_gen_tokens, self.max_length - 1)
            prompt = self.tokenizer.encode(context, add_bos=True)
            prompt = prompt[-(self.max_length - max_new_tokens) :]
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
            temperature = float(kwargs.get("temperature", 0.0))
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=kwargs.get("top_k"),
                top_p=float(kwargs.get("top_p", 1.0)),
                eos_id=self.tokenizer.eos_id,
            )
            generated = self.tokenizer.decode(output[0, len(prompt) :].tolist())
            stop_positions = [generated.find(stop) for stop in stop_strings if stop in generated]
            if stop_positions:
                generated = generated[: min(stop_positions)]
            results.append(generated)
        return results


def harness_settings(*, include_gsm8k: bool = False) -> dict[str, Any]:
    """Return the exact task/version settings persisted beside benchmark results."""

    tasks = [*DEFAULT_TASKS]
    if include_gsm8k:
        tasks.extend(OPTIONAL_TASKS)
    try:
        version = importlib.metadata.version("lm-eval")
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    return {
        "lm_eval_version": version,
        "tasks": tasks,
        "fewshot": 0,
        "apply_chat_template": False,
        "log_samples": True,
    }
