"""lm-evaluation-harness adapter for MiniFrontier (MF-035).

Beginner's map of this file
---------------------------
``lm-eval`` is the community's standard suite of multiple-choice-style benchmarks
(HellaSwag, ARC, and friends). Most of those tasks do not ask the model to *write*
anything -- they score each candidate answer by how likely the model thinks it is,
and pick the winner. This adapter is the translation layer that lets an outside
harness ask a MiniFrontier model those likelihood questions.

Worth keeping in perspective: a 150M-parameter model scores near chance on most
of these. The value here is having a comparable, standard number rather than a
flattering one.
"""

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
# MF-086 part (3): arc_easy/hellaswag/piqa alone score near chance at this
# project's scale and would not detect most of the improvements proposed in
# MF-081/082. Kept opt-in (like OPTIONAL_TASKS) rather than folded into
# DEFAULT_TASKS -- "blimp" alone expands to 67 real BLiMP paradigm subtasks
# (verified against the installed lm-eval's own task registry), so appending
# these unconditionally would multiply the cost of every quick default-task
# smoke/comparison run across this project, not just the real evaluation
# passes these tasks actually exist for.
EXTENDED_TASKS = (
    "blimp",
    "lambada_openai",
    "winogrande",
    "openbookqa",
    "commonsense_qa",
    "boolq",
)
# MF-122: CRUXEval (Gu et al., 2024, cited directly in StarCoder2's own
# paper) -- CRUXEval-I asks the model to predict an input that makes a
# given Python function produce a given output; CRUXEval-O asks it to
# predict the function's output for a given input. Genuinely more targeted
# at code reasoning/execution than anything in DEFAULT_TASKS/EXTENDED_TASKS,
# none of which execute code at all. Kept in its own list rather than
# folded into EXTENDED_TASKS, for one real, disclosed reason: scoring it
# means actually EXECUTING the model's generated code -- lm-eval's own
# installed task config marks it `unsafe_code: true`, and the harness
# itself refuses to run an unsafe-code task unless the caller passes
# `confirm_run_unsafe_code=True` to `simple_evaluate` (see
# scripts/eval.py/scripts/eval_checkpoint.py's own --include-cruxeval
# wiring). That execution is real `exec()` inside a plain subprocess with
# no sandbox beyond a resource-limiting guard (lm-eval's own
# tasks/cruxeval/utils.py, the same "not a security sandbox" caveat this
# project's own evaluation/code.py already discloses for score_python's
# fixture execution) -- never run without explicit, informed opt-in, never
# bundled into a default/extended sweep a caller might not expect to
# execute arbitrary model output. Only the two base (non-chain-of-thought,
# non-0.8-temperature) variants are included; `cruxeval_input_cot`/
# `cruxeval_output_cot`/`cruxeval_input_08`/`cruxeval_output_08` also exist
# in the installed lm-eval but are outside what MF-122 asked for.
CODE_EXECUTION_TASKS = ("cruxeval_input", "cruxeval_output")


class MiniFrontierEvalLM(LM):
    """Correctness-first single-device adapter for the standard harness API."""

    def __init__(
        self,
        model: MiniFrontier,
        tokenizer: MiniFrontierTokenizer,
        *,
        max_gen_tokens: int = 64,
        eval_batch_size: int = 1,
    ) -> None:
        super().__init__()
        if max_gen_tokens <= 0:
            raise ValueError("max_gen_tokens must be positive")
        if eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_gen_tokens = max_gen_tokens
        # MF-114: how many loglikelihood/loglikelihood_rolling requests to
        # fold into one forward call (see _score_batch below). Defaults to 1
        # -- the original, unbatched behavior -- so any existing caller that
        # does not pass this explicitly sees no change at all; scripts/eval.py
        # and scripts/eval_checkpoint.py expose it as --harness-batch-size.
        self.eval_batch_size = eval_batch_size
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

    def _score(self, prefix: list[int], continuation: list[int]) -> tuple[float, bool]:
        """Log-probability of ``continuation`` given ``prefix``, teacher-forced.

        A thin single-pair call into ``_score_batch`` below -- kept as its own
        method because every existing caller/test names it directly, but the
        real logic (and the real correctness guarantee that batched and
        unbatched scoring agree exactly) now lives in one place, not two.
        """

        return self._score_batch([(prefix, continuation)])[0]

    @torch.inference_mode()
    def _score_batch(self, pairs: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        """Log-probability of each (prefix, continuation) pair, batched into
        as few forward calls as possible (MF-114).

        The whole sequence is already fully known for every pair -- nothing
        here is generated or sampled -- so, exactly like the original
        single-pair ``_score``, this never needs a KV-cache or a token-by-token
        loop: one forward pass already returns logits at every position.

        Right-padding shorter sequences up to the batch's own max length is
        the key trick, and it is safe with *zero* new masking code because of
        how causal attention already works here: every mask this project
        builds (``masking.build_attention_mask``) only ever allows a query
        position to attend to key positions at or before it. A padding token
        placed strictly *after* a row's real content can therefore never be
        seen by any of that row's real positions, regardless of what token id
        the padding actually is -- and since every per-token operation in the
        model (embedding, RMSNorm, attention within a row, SwiGLU, the output
        head) is otherwise batch-independent, a shorter row's own real result
        is completely unaffected by a longer row sharing its batch. Each row's
        real logits are read off at that row's own real length, never the
        batch's padded length, so this is not an approximation -- it is
        byte-for-byte the same computation ``_score`` would do alone, just
        sharing one forward call across rows that happen to fit together.

        A pair whose own (prefix, continuation) already exceeds
        ``max_length`` is scored individually via the slow sliding-window
        fallback, exactly as the unbatched path already did -- that fallback
        was never part of what this batches, only the common, single-forward
        fast path is.
        """

        results: list[tuple[float, bool] | None] = [None] * len(pairs)
        batchable_indices: list[int] = []
        full_sequences: list[list[int]] = []
        for index, (prefix, continuation) in enumerate(pairs):
            if not continuation:
                results[index] = (0.0, True)
                continue
            full_sequence = [self.tokenizer.bos_id, *prefix, *continuation]
            if len(full_sequence) > self.max_length:
                results[index] = self._score_sliding_window(prefix, continuation)
                continue
            batchable_indices.append(index)
            full_sequences.append(full_sequence)

        if full_sequences:
            max_len = max(len(sequence) for sequence in full_sequences)
            padded = torch.full(
                (len(full_sequences), max_len),
                self.tokenizer.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, sequence in enumerate(full_sequences):
                padded[row, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=self.device
                )
            logits = self.model(padded).logits.float()  # [B, Smax, vocab]
            for row, index in enumerate(batchable_indices):
                continuation = pairs[index][1]
                continuation_length = len(continuation)
                real_length = len(full_sequences[row])
                # Same slice _score always used, just anchored to this row's
                # own real length instead of a fixed offset from the end.
                predicting_logits = logits[
                    row, real_length - continuation_length - 1 : real_length - 1
                ]
                targets = torch.tensor(continuation, dtype=torch.long, device=self.device)
                log_probs = torch.log_softmax(predicting_logits, dim=-1)
                token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                is_greedy = bool((predicting_logits.argmax(dim=-1) == targets).all())
                results[index] = (float(token_log_probs.sum().cpu()), is_greedy)

        return results  # type: ignore[return-value]

    @torch.inference_mode()
    def _score_sliding_window(
        self, prefix: list[int], continuation: list[int]
    ) -> tuple[float, bool]:
        """Rare fallback: (prefix, continuation) exceeds ``max_length``.

        Kept as the original per-token loop rather than folded into the fast
        path above, since the fast path's single fixed truncation window
        would give *earlier* continuation tokens less prefix context than
        this sliding window does (it drops the oldest tokens one at a time as
        the loop progresses, rather than all at once) -- a real behavior
        difference, not just a speed one, when truncation actually triggers.
        Every real task this project evaluates uses short (sentence-length)
        examples well under any real ``max_length``, so this path is not
        expected to run in practice; it exists so a future oversized example
        degrades to the slow-but-correct original path instead of silently
        changing what gets scored.
        """

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
        results: list[tuple[float, bool]] = []
        for start in range(0, len(requests), self.eval_batch_size):
            chunk = requests[start : start + self.eval_batch_size]
            pairs = [
                (self.tokenizer.encode(context), self.tokenizer.encode(continuation))
                for context, continuation in (request.args for request in chunk)
            ]
            results.extend(self._score_batch(pairs))
        return results

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        results: list[float] = []
        for start in range(0, len(requests), self.eval_batch_size):
            chunk = requests[start : start + self.eval_batch_size]
            pairs = [([], self.tokenizer.encode(request.args[0])) for request in chunk]
            results.extend(log_prob for log_prob, _ in self._score_batch(pairs))
        return results

    @torch.inference_mode()
    def generate_until(self, requests: list[Any]) -> list[str]:
        """MF-114 scope note: deliberately NOT batched, unlike ``loglikelihood``/
        ``loglikelihood_rolling`` above. Those are safe to batch with zero new
        masking code because right-padding only ever adds tokens *after* the
        real content being scored, which causal attention can never look
        forward into. Generation is the opposite shape: a shorter prompt
        padded up to a batch's longest prompt would need its padding placed
        *before* the point where new tokens get generated, and those padding
        tokens would then sit in the causal past of every generated token --
        real contamination, not a performance-only change. Fixing that needs
        genuine per-row key-padding-mask support in the model's forward/
        attention path (``masking.py`` only builds one mask shared by the
        whole batch today), a materially bigger change than this task's real
        scope -- left as real, disclosed future work, not silently assumed
        solved by the same trick used above.
        """

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


def harness_settings(
    *,
    include_gsm8k: bool = False,
    include_extended: bool = False,
    include_cruxeval: bool = False,
) -> dict[str, Any]:
    """Return the exact task/version settings persisted beside benchmark results.

    ``include_cruxeval`` doubles as this project's own explicit consent to
    CRUXEval's real code execution (see ``CODE_EXECUTION_TASKS``'s own
    comment) -- the returned ``confirm_run_unsafe_code`` flag is what a
    caller must thread into ``simple_evaluate`` for the task to actually
    run at all; lm-eval refuses otherwise. Every task here, CRUXEval
    included, still runs at this project's own established 0-shot
    convention (``fewshot: 0`` below, applied globally by every real
    caller) -- a real, disclosed departure from CRUXEval's own suggested
    2-shot setup (its installed task YAML's `num_fewshot: 2`), not an
    oversight: this project already evaluates every other task (gsm8k
    included) at 0-shot regardless of its own suggested convention, for one
    consistent, comparable number across the whole suite.
    """

    tasks = [*DEFAULT_TASKS]
    if include_gsm8k:
        tasks.extend(OPTIONAL_TASKS)
    if include_extended:
        tasks.extend(EXTENDED_TASKS)
    if include_cruxeval:
        tasks.extend(CODE_EXECUTION_TASKS)
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
        "confirm_run_unsafe_code": include_cruxeval,
    }
