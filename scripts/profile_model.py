"""Profile a labeled single-stream MiniFrontier prefill/decode path.

Where does the time and memory actually go? This measures the two phases of
generation separately, because they behave nothing alike:

* **Prefill** -- the whole prompt in one pass. Lots of parallel work, compute
  bound, and it is what fills the KV cache.
* **Decode** -- one token at a time. Very little arithmetic per step, so it is
  bound by memory bandwidth instead: nearly all the time goes on reading the
  weights and the cache, not on the multiply-adds.

That difference explains a lot of LLM engineering. It is why the KV cache matters
so much, why batching helps decode more than prefill, and why "tokens per second"
is meaningless without saying which phase you measured.

The filename deliberately avoids ``profile.py``, which shadows Python's standard-library
``profile`` module when other scripts import PyTorch compilation/determinism machinery.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from minifrontier.cache import KVCache
from minifrontier.compilation import maybe_compile
from minifrontier.config import ModelConfig
from minifrontier.model import MiniFrontier
from minifrontier.precision import cast_model_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--context-override", type=int)
    parser.add_argument("--window-override", type=int)
    parser.add_argument("--performance-only-override", action="store_true")
    parser.add_argument("--compile-prefill", action="store_true")
    parser.add_argument("--compile-backend")
    parser.add_argument(
        "--full-history-local-cache",
        action="store_true",
        help="Use the correctness-reference cache instead of bounded local ring storage.",
    )
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.prompt_length <= 0 or args.decode_tokens <= 0:
        raise ValueError("batch size, prompt length, and decode tokens must be positive")
    config = ModelConfig.from_toml(args.config)
    if args.context_override is not None or args.window_override is not None:
        if not args.performance_only_override:
            raise ValueError("context/window overrides require --performance-only-override")
        context = args.context_override or config.max_seq_len
        window = args.window_override or config.local_window
        config = replace(config, max_seq_len=context, local_window=window)
    if args.prompt_length + args.decode_tokens > config.max_seq_len:
        raise ValueError("prompt plus decode tokens exceeds configured context")
    device = torch.device(args.device)
    report: dict[str, object] = {
        "status": "failed",
        "quality_claim": False,
        "performance_only_override": args.performance_only_override,
        "config": config.to_dict(),
        "device": str(device),
        "device_type": device.type,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "selected_attention_impl": config.attention_impl,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    try:
        model = MiniFrontier(config)
        policy = cast_model_for_inference(model, args.precision, device)
        prefill_model, compile_report = maybe_compile(
            model,
            enabled=args.compile_prefill,
            path="prefill",
            backend=args.compile_backend,
        )
        tokens = torch.randint(
            0,
            config.vocab_size,
            (args.batch_size, args.prompt_length),
            device=device,
        )
        cache = KVCache.allocate(
            config,
            batch_size=args.batch_size,
            device=device,
            capacity=args.prompt_length + args.decode_tokens,
            bounded_local=(
                config.attention_pattern == "hybrid" and not args.full_history_local_cache
            ),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = prefill_model(tokens, cache=cache, logits_to_keep=1).logits[:, 0]
        _sync(device)
        prefill_seconds = time.perf_counter() - started
        decode_latencies = []
        for _ in range(args.decode_tokens):
            next_token = logits.argmax(dim=-1, keepdim=True)
            _sync(device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(next_token, cache=cache, logits_to_keep=1).logits[:, 0]
            _sync(device)
            decode_latencies.append(time.perf_counter() - started)
        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        total_vram = (
            torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
        )
        report.update(
            {
                "status": "completed",
                "precision": policy.resolved,
                "cache_dtype": str(cache.layers[0].dtype),
                "compile": asdict(compile_report),
                "time_to_first_token_seconds": prefill_seconds,
                "prefill_tokens_per_second": args.batch_size * args.prompt_length / prefill_seconds,
                "mean_inter_token_seconds": sum(decode_latencies) / len(decode_latencies),
                "decode_tokens_per_second": args.batch_size
                * len(decode_latencies)
                / sum(decode_latencies),
                "cache_allocated_bytes": cache.allocated_bytes(),
                "cache_logical_bytes": cache.logical_bytes(),
                "bounded_local_cache": any(layer.ring for layer in cache.layers),
                "peak_allocated_vram_bytes": peak_allocated,
                "peak_reserved_vram_bytes": peak_reserved,
                "total_vram_bytes": total_vram,
                "peak_reserved_vram_fraction": (
                    peak_reserved / total_vram if total_vram else "unmeasured"
                ),
                "scope": "single_stream_or_fixed_batch",
            }
        )
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({report['status']})")


if __name__ == "__main__":
    main()
