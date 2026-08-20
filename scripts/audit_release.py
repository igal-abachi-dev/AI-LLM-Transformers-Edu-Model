"""Load-test and audit matched MiniFrontier Edu/Modern release directories."""

# The last gate before publishing. Loads each release exactly as a stranger with a
# fresh clone would, checks every file against the SHA-256 manifest, and confirms
# the Edu and Modern pair really are matched -- same tokenizer, same budget, same
# evaluation -- since "matched" is the only thing that makes comparing them fair.
#
# Failures here are the boring, expensive kind: a missing file, a stale hash, a
# model that only loads because of something left over in your working directory.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minifrontier.release import audit_release_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edu", type=Path, required=True)
    parser.add_argument("--modern", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_release_pair(args.edu, args.modern)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
