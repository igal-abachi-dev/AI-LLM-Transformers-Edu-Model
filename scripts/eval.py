"""Evaluate a MiniFrontier release on local validation text and optional lm-eval tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from minifrontier.checkpoint import load_release
from minifrontier.evaluation.language import MiniFrontierEvalLM, harness_settings
from minifrontier.evaluation.validation import batches_from_texts, evaluate_token_batches
from minifrontier.precision import cast_model_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--run-harness", action="store_true")
    parser.add_argument("--include-gsm8k", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def _validation_texts(path: Path, field: str) -> list[str]:
    texts = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if field not in row:
            raise ValueError(f"missing {field!r} at {path}:{line_number}")
        texts.append(str(row[field]))
    return texts


def _adapter_smoke(adapter: MiniFrontierEvalLM) -> dict[str, Any]:
    likelihood = adapter.loglikelihood([SimpleNamespace(args=("Mini", "Frontier"))])[0]
    generated = adapter.generate_until(
        [SimpleNamespace(args=("Mini", {"max_gen_toks": 2, "until": []}))]
    )[0]
    return {
        "status": "completed",
        "loglikelihood": likelihood[0],
        "is_greedy": likelihood[1],
        "generated_utf8_bytes": len(generated.encode("utf-8")),
    }


def _json_default(value: Any) -> object:
    """Preserve scalar metrics from NumPy/Torch and stringify harness metadata objects."""

    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


def main() -> None:
    args = parse_args()
    model, tokenizer = load_release(args.release, device=args.device)
    policy = cast_model_for_inference(model, args.precision, args.device)
    adapter = MiniFrontierEvalLM(model, tokenizer)
    report: dict[str, Any] = {
        "settings": harness_settings(include_gsm8k=args.include_gsm8k),
        "adapter_smoke": _adapter_smoke(adapter),
        "validation": None,
        "harness": {"status": "not_requested"},
        "precision": asdict(policy),
    }
    if args.validation_jsonl is not None:
        texts = _validation_texts(args.validation_jsonl, args.text_field)
        batches = batches_from_texts(
            tokenizer,
            texts,
            max_seq_len=model.config.max_seq_len,
            device=args.device,
        )
        report["validation"] = asdict(
            evaluate_token_batches(model, batches, pad_id=tokenizer.pad_id)
        )
    if args.run_harness:
        from lm_eval import simple_evaluate

        try:
            report["harness"] = {
                "status": "completed",
                "results": simple_evaluate(
                    model=adapter,
                    tasks=report["settings"]["tasks"],
                    num_fewshot=0,
                    limit=args.limit,
                    log_samples=True,
                ),
            }
        except Exception as error:  # preserve infrastructure failure separately from scores
            report["harness"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
