#!/usr/bin/env python3
"""Compare two validation-only research summaries and emit a diagnosis JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.failure_analysis import diagnose


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only validation-result diagnosis; never loads CIFAR-10.")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="Optional append-only diagnosis artifact.")
    args = parser.parse_args()
    candidate, control = json.loads(args.candidate.read_text()), json.loads(args.control.read_text())
    result = diagnose(candidate, control)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite diagnosis evidence: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
