#!/usr/bin/env python3
"""Check that the human-readable hypothesis registry has required fields."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "RESEARCH_HYPOTHESES.md").read_text()
REQUIRED_IDS = {f"H{number}" for number in range(1, 11)}


def main() -> None:
    found = set(re.findall(r"^## (H\d+):", TEXT, flags=re.MULTILINE))
    missing = REQUIRED_IDS - found
    if missing:
        raise SystemExit(f"Missing hypothesis IDs: {sorted(missing)}")
    for hypothesis in sorted(REQUIRED_IDS, key=lambda value: int(value[1:])):
        section = re.search(rf"^## {hypothesis}:.*?(?=^## H\d+:|\Z)", TEXT, flags=re.MULTILINE | re.DOTALL)
        if section is None or any(token not in section.group(0) for token in ("**Status:**", "**Falsification:", "**Linked experiments:**")):
            raise SystemExit(f"{hypothesis} lacks required status, falsification, or experiment linkage.")
    print("Hypothesis registry validation passed; no mutation performed.")


if __name__ == "__main__":
    main()
