"""Create OOM-safe 350M/500M accounting and CPU wiring evidence without GPU claims."""

# Before training a bigger model, work out on paper whether it can possibly fit:
# parameters, optimizer state, activations and KV cache against the memory the GPU
# actually has. Arithmetic is cheap; discovering an out-of-memory error forty
# minutes into a run is not.
#
# It deliberately produces accounting and wiring evidence only. Nothing here is a
# measured GPU result, and the file name says so.

from __future__ import annotations

import argparse
from pathlib import Path

from minifrontier.config import ModelConfig
from minifrontier.scale import ScaleStopCriteria, build_preflight_report, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-350m", type=Path, default=Path("configs/350m-modern.toml"))
    parser.add_argument("--config-500m", type=Path, default=Path("configs/500m-modern.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--cpu-checkpoint-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {"350m-modern": args.config_350m, "500m-modern": args.config_500m}
    report = build_preflight_report(
        {name: ModelConfig.from_toml(path) for name, path in paths.items()},
        config_paths=paths,
        batch_size=args.batch_size,
        context_length=args.context_length,
        run_cpu_smoke=args.cpu_checkpoint_smoke,
        stop_criteria=ScaleStopCriteria(),
    )
    write_json_report(args.output, report)
    print(f"wrote preflight-only report to {args.output}; CUDA fields remain unmeasured")


if __name__ == "__main__":
    main()
