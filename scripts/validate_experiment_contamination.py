#!/usr/bin/env python3
"""Validate whether a candidate declares only approved changes from its parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTECTED = {
    "dataset_version", "dataset_split_hash", "teacher_checkpoint_hash",
    "student_architecture", "student_initialization", "qat_baseline",
    "quantizer", "seed_protocol",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only experiment-comparability check; no datasets are loaded.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    parent, candidate = json.loads(args.parent.read_text()), json.loads(args.candidate.read_text())
    declared_changed = set(candidate.get("changed_variables", []))
    conflicts = []
    for key in sorted(PROTECTED):
        if key in declared_changed:
            conflicts.append(f"protected variable declared changed: {key}")
        elif parent.get(key) is not None and candidate.get(key) is not None and parent[key] != candidate[key]:
            conflicts.append(f"undeclared protected mismatch: {key}")
    for key in candidate.get("fixed_variables", []):
        if parent.get(key) is not None and candidate.get(key) is not None and parent[key] != candidate[key]:
            conflicts.append(f"declared fixed variable differs from parent: {key}")
    result = {
        "parent_experiment": parent.get("experiment_id"), "candidate_experiment": candidate.get("experiment_id"),
        "status": "CONTAMINATED" if conflicts else "COMPARABLE", "conflicts": conflicts,
        "required_declarations": ["parent_control", "hypothesis_id", "mechanism", "changed_variables", "fixed_variables"],
        "test_evaluation": "not_run",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite contamination evidence: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(encoded)
    print(encoded, end="")
    if conflicts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
