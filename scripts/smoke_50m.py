"""Run one real-text 50M Edu optimization step and verify the artifact round trip."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from minifrontier.cache import KVCache
from minifrontier.checkpoint import load_training_checkpoint, save_training_checkpoint
from minifrontier.config import ModelConfig
from minifrontier.data import Document, filter_and_deduplicate, pack_documents, split_documents
from minifrontier.evaluation.benchmark import (
    BenchmarkRecord,
    ComparisonKey,
    hardware_description,
    measure_forward_throughput,
    write_record,
)
from minifrontier.evaluation.code import (
    assert_no_contamination,
    load_fixtures,
    normalized_hash,
)
from minifrontier.evaluation.language import harness_settings
from minifrontier.evaluation.validation import batches_from_texts, evaluate_token_batches
from minifrontier.model import MiniFrontier
from minifrontier.reproducibility import seed_everything
from minifrontier.tokenizer import train_byte_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/50m-edu.toml"))
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[Path("plan.md"), Path("more-context.md")],
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("reports/50m-edu-cpu-smoke.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/50m-edu-cpu-smoke"))
    return parser.parse_args()


def _real_documents(paths: list[Path]) -> list[Document]:
    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        paragraphs = [part.strip() for part in text.split("\n\n") if len(part.strip()) >= 64]
        for index, paragraph in enumerate(paragraphs):
            documents.append(
                Document.create(
                    paragraph,
                    source="MiniFrontier repository prose",
                    revision="working-tree-2026-08-17",
                    license="Apache-2.0",
                    language="en",
                    record_id=f"{path.as_posix()}:{index}",
                    path=path.as_posix(),
                )
            )
    return list(filter_and_deduplicate(documents))


def _write_scorecard(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    forward_summary = (
        f"Measured CPU forward tokens/s ({metrics['forward_iterations']} iterations): "
        f"{metrics['forward_tokens_per_second']:.3f}"
    )
    markdown = f"""# 50M Edu CPU smoke scorecard

This is an **engineering baseline**, not a model-quality claim. It is one FP32 optimization
step over packed, project-authored prose on a CPU-only Azure Dev Box. The planned 1-5M-token
FineWeb-Edu smoke and standard lm-eval scores remain GPU work for the home RTX machine.

## Result

- Parameters: {report["parameters"]:,}
- Packed train tokens: {report["train_tokens"]}
- Loss after the step: {metrics["train_loss"]:.6f}
- Validation cross-entropy: {metrics["cross_entropy"]:.6f}
- Validation perplexity: {metrics["perplexity"]:.6f}
- Validation bits/byte: {metrics["bits_per_byte"]:.6f}
- {forward_summary}
- Allocated KV-cache bytes: {report["kv_cache_bytes"]:,}
  (batch 1, capacity {report["cache_capacity"]})
- Checkpoint logits exact after reload: {report["checkpoint_exact"]}
- Harness adapter smoke: {report["lm_eval_adapter_status"]}

## Limitations

- One step cannot establish convergence, downstream quality, or useful generation.
- CPU throughput is not comparable to the future RTX BF16/SDPA/compile measurements.
- ARC-Easy, HellaSwag, PIQA, and optional GSM8K are configured but not downloaded here.
- Raw comparable record: `{report["benchmark_record"]}`.
"""
    path.with_suffix(".md").write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence-length must be at least two")
    seed_everything(args.seed)
    started = time.perf_counter()
    config = ModelConfig.from_toml(args.config)
    documents = _real_documents(args.source)
    if len(documents) < 2:
        raise ValueError("at least two real-text documents are required")
    fixture_path = Path("eval/fixtures/code_fim_v1.jsonl")
    fixtures = load_fixtures(fixture_path)
    assert_no_contamination(fixtures, {normalized_hash(item.text) for item in documents})
    train_documents, validation_documents = split_documents(
        documents,
        validation_fraction=0.1,
    )
    if not train_documents or not validation_documents:
        train_documents, validation_documents = documents[:-1], documents[-1:]

    tokenizer = train_byte_bpe(
        (document.text for document in train_documents),
        vocab_size=config.vocab_size,
        min_frequency=2,
    )
    tokenizer_path = args.artifacts / "tokenizer"
    tokenizer.save(tokenizer_path)
    packed = list(
        pack_documents(
            train_documents,
            tokenizer,
            sequence_length=args.sequence_length,
            drop_remainder=False,
        )
    )
    if not packed:
        raise RuntimeError("real-text packing produced no sequences")
    train_tokens = packed[0].tensor().unsqueeze(0)

    model = MiniFrontier(config)
    if model.parameter_count() != 53_361_152:
        raise RuntimeError(f"50M config drifted to {model.parameter_count():,} parameters")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    step_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(train_tokens, labels=train_tokens)
    assert output.loss is not None
    if not torch.isfinite(output.loss):
        raise RuntimeError("non-finite smoke loss")
    output.loss.backward()
    if not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise RuntimeError("non-finite smoke gradients")
    optimizer.step()
    step_seconds = time.perf_counter() - step_started
    train_loss = float(output.loss.detach())

    model.eval()
    with torch.inference_mode():
        expected_logits = model(train_tokens).logits.detach().clone()
    checkpoint_path = args.artifacts / "checkpoint"
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        trainer_state={"step": 1, "train_tokens": train_tokens.numel()},
        data_cursor={"packed_sequence": 1},
    )
    del output, optimizer, model
    gc.collect()

    reloaded = MiniFrontier(config)
    trainer_state, data_cursor = load_training_checkpoint(
        checkpoint_path,
        reloaded,
        restore_rng=False,
        trusted_local_state=True,
    )
    reloaded.eval()
    with torch.inference_mode():
        actual_logits = reloaded(train_tokens).logits
    checkpoint_exact = torch.equal(expected_logits, actual_logits)
    if not checkpoint_exact or trainer_state["step"] != 1 or data_cursor["packed_sequence"] != 1:
        raise RuntimeError("checkpoint round trip failed")

    validation_text = validation_documents[0].text[:256]
    validation_batches = batches_from_texts(
        tokenizer,
        [validation_text],
        max_seq_len=config.max_seq_len,
    )
    validation = evaluate_token_batches(reloaded, validation_batches, pad_id=tokenizer.pad_id)
    forward_iterations = 5
    forward_tps, forward_seconds, peak_vram = measure_forward_throughput(
        reloaded,
        train_tokens,
        iterations=forward_iterations,
    )
    cache = KVCache.allocate(
        config,
        batch_size=1,
        device="cpu",
        dtype=next(reloaded.parameters()).dtype,
        capacity=args.sequence_length + 2,
    )
    generated = reloaded.generate(
        train_tokens[:, : min(8, train_tokens.shape[1])],
        max_new_tokens=2,
        temperature=0.0,
        eos_id=tokenizer.eos_id,
    )
    if generated.shape[1] < train_tokens[:, :8].shape[1]:
        raise RuntimeError("generation shortened the prompt")

    tokenizer_hash = json.loads(
        (tokenizer_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )["tokenizer_sha256"]
    data_id = hashlib.sha256(
        "".join(document.content_hash for document in documents).encode("ascii")
    ).hexdigest()
    comparison = ComparisonKey(
        data_id=data_id,
        tokenizer_hash=tokenizer_hash,
        token_budget=train_tokens.numel(),
        batch_tokens=train_tokens.numel(),
        context_length=args.sequence_length,
        seed_policy=f"torch/numpy/python={args.seed}",
        evaluation_id="validation-v1+lm-eval-0.4.12-suite-config",
    )
    record_path = Path("reports/runs/50m-edu-cpu-smoke.json")
    record = BenchmarkRecord(
        run_id="50m-edu-cpu-smoke-2026-08-17",
        comparison=comparison,
        quality={
            "cross_entropy": validation.cross_entropy,
            "perplexity": validation.perplexity,
            "bits_per_byte": validation.bits_per_byte,
        },
        training_tokens_per_second=train_tokens.numel() / step_seconds,
        inference_tokens_per_second=forward_tps,
        wall_time_seconds=time.perf_counter() - started,
        peak_vram_bytes=peak_vram,
        kv_cache_bytes=cache.allocated_bytes(),
        hardware=hardware_description(),
        notes=[
            "One-step FP32 CPU engineering smoke; no model-quality claim.",
            "Standard lm-eval datasets not downloaded on the Azure Dev Box.",
        ],
    )
    write_record(record, record_path)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "hardware": hardware_description(),
        "config": str(args.config),
        "parameters": reloaded.parameter_count(),
        "documents": len(documents),
        "evaluation_fixture_contamination": "none",
        "evaluation_fixture_path": fixture_path.as_posix(),
        "packed_sequences": len(packed),
        "train_tokens": train_tokens.numel(),
        "cache_capacity": cache.capacity,
        "kv_cache_bytes": cache.allocated_bytes(),
        "checkpoint_exact": checkpoint_exact,
        "generated_tokens": generated.shape[1] - min(8, train_tokens.shape[1]),
        "lm_eval_adapter_status": "locally unit-tested; standard tasks not run",
        "lm_eval_settings": harness_settings(),
        "benchmark_record": record_path.as_posix(),
        "metrics": {
            "train_loss": train_loss,
            "cross_entropy": validation.cross_entropy,
            "perplexity": validation.perplexity,
            "bits_per_byte": validation.bits_per_byte,
            "forward_tokens_per_second": forward_tps,
            "forward_seconds": forward_seconds,
            "forward_iterations": forward_iterations,
            "training_step_seconds": step_seconds,
        },
        "limitations": record.notes,
    }
    _write_scorecard(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
