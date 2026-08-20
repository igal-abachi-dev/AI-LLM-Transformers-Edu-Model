"""Exercise vLLM's OpenAI completions/chat transport and retain the raw responses."""

# vLLM is a high-throughput serving engine that speaks the OpenAI API, so ordinary
# client libraries can talk to a locally served model. This checks that transport
# works and keeps the raw responses as evidence.
#
# Note what a successful connection does NOT prove. The API answering is a
# transport fact; whether the served model produces the same tokens as the native
# implementation is a separate parity question, and only that earns a
# compatibility claim.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.ecosystem import validate_vllm_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="local-key")
    parser.add_argument("--prompt", default="Write a Python function that adds two integers.")
    parser.add_argument(
        "--system-prompt", default="You are MiniFrontier, a helpful coding assistant."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-fixture", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {"transport_status": "failed"}
    try:
        report = validate_vllm_api(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            parity_fixture=(
                json.loads(args.parity_fixture.read_text(encoding="utf-8"))
                if args.parity_fixture
                else None
            ),
        )
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote vLLM transport status={report['transport_status']} to {args.output}")
    if report["transport_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
