"""Create a Q4_K_M candidate while refusing to label it publish-ready without evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.gguf import quantize_q4_k_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-quantize", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-cpp-revision", required=True)
    parser.add_argument("--source-model-revision", required=True)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = (
        json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
        if args.calibration_manifest
        else None
    )
    report = quantize_q4_k_m(
        llama_quantize=args.llama_quantize,
        high_precision_gguf=args.source,
        output=args.output,
        llama_cpp_revision=args.llama_cpp_revision,
        source_model_revision=args.source_model_revision,
        calibration_manifest=calibration,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote Q4_K_M candidate to {args.output}; publish_ready=false pending GPU/evaluation")


if __name__ == "__main__":
    main()
