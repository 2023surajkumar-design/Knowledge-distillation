#!/usr/bin/env python3
"""Fail-closed validation for the append-only Task-4 completion protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/kd/resnet18_ternary_vanillaKD_final_rerun1.yaml"
CONT = ROOT / "configs/kd/resnet18_ternary_vanillaKD_remaining_rerun1.yaml"
HISTORY = ROOT / "experiments/task4/histories/task4_b3_final_t2_lam09_rerun1/seed42.json"
CHECKPOINT = ROOT / "experiments/task4/checkpoints/task4_b3_final_t2_lam09_rerun1/resnet18_ternary_kd_seed42.pth"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    base, cont = yaml.safe_load(BASE.read_text()), yaml.safe_load(CONT.read_text())
    assert HISTORY.is_file() and CHECKPOINT.is_file(), "seed-42 preservation artifacts are absent"
    assert base["seeds"] == [42, 43, 44] and cont["seeds"] == [43, 44]
    mutable = {"run_name", "condition", "seeds"}
    mismatches = {k: (base.get(k), cont.get(k)) for k in set(base) | set(cont)
                  if k not in mutable and base.get(k) != cont.get(k)}
    assert not mismatches, f"continuation changes frozen Task-4 protocol: {mismatches}"
    history = json.loads(HISTORY.read_text())
    assert history["config"]["seed"] == 42
    assert history["best_validation_accuracy"] == 0.9514
    assert history["config"]["temperature"] == cont["temperature"]
    assert history["config"]["kd_lambda"] == cont["kd_lambda"]
    assert history["config"]["quant_config"] == cont["quant"]
    print(json.dumps({"status": "PASS", "completed_seed": 42,
                      "checkpoint_sha256": sha256(CHECKPOINT),
                      "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
                      "remaining_seeds": cont["seeds"],
                      "test_evaluation": "not_run"}, indent=2))

if __name__ == "__main__":
    main()
