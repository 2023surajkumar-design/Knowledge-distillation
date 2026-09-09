#!/usr/bin/env python3
"""Validate the append-only experiment ledger without mutating it."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "research_db" / "schema.json").read_text())
ENTRIES = ROOT / "research_db" / "experiments.jsonl"


def main() -> None:
    required = set(SCHEMA["required"])
    version_two = set(SCHEMA["notes"]["required_for_schema_version_2"])
    errors: list[str] = []
    for line_number, raw in enumerate(ENTRIES.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        entry = json.loads(raw)
        missing = sorted(required - entry.keys())
        if missing:
            errors.append(f"line {line_number}: missing {missing}")
        if entry.get("result") not in {"pending", "completed", "failed", "inconclusive"}:
            errors.append(f"line {line_number}: invalid result {entry.get('result')!r}")
        if entry.get("schema_version") == 2:
            missing_v2 = sorted(version_two - entry.keys())
            if missing_v2:
                errors.append(f"line {line_number}: schema-v2 missing {missing_v2}")
    if errors:
        raise SystemExit("Research DB validation failed:\n" + "\n".join(errors))
    print("Research DB validation passed; no mutation performed.")


if __name__ == "__main__":
    main()
