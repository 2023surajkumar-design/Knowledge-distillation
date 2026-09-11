#!/usr/bin/env python3
"""Persistent artifact-first controller for final validation-only reporting.

All currently authorized training is complete. This controller intentionally
does not relaunch Task 4 or DKD, and it never accesses the official test set.
Future one-seed training must be added as a separately reviewed stage.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "research_state_checkpoint.json"
SUMMARY_PATHS = {
    "T4": ROOT / "results/task4/task4_b3_final_t2_lam09_remaining_rerun1_summary.json",
    "T4_remaining": ROOT / "results/task4/task4_b3_final_t2_lam09_remaining_rerun1_summary.json",
    "DKD": ROOT / "results/task4/task5_dkd_final_t2_lam09_a1_b8_r1_summary.json",
    "DIST_smoke": ROOT / "results/task4/task6_dist_smoke_r1_summary.json",
    "DIST": ROOT / "results/task4/task6_dist_screen_t2_lam09_i1_a1_r1_summary.json",
}

def read_json(path: Path) -> dict:
    return json.loads(path.read_text())

def checkpoint_state(**extra: object) -> None:
    state = read_json(STATE) if STATE.exists() else {}
    state.update({"timestamp": datetime.now().astimezone().isoformat(),
                  "run_mode": "TIME_CONSTRAINED_UNIFIED_NOTEBOOK_EXECUTION",
                  "test_evaluation": "LOCKED_NOT_RUN",
                  "automl_status": "PAUSED_NOT_ABANDONED", **extra})
    STATE.write_text(json.dumps(state, indent=2) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    missing = [name for name, path in SUMMARY_PATHS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preserved summaries: {missing}")
    summaries = {name: read_json(path) for name, path in SUMMARY_PATHS.items()}
    if any(data.get("test_evaluation") != "not_run" for data in summaries.values()):
        raise RuntimeError("Validation-only controller found a test-evaluation record.")
    evidence = {
        "status": "COMPLETE_VALIDATION_ONLY",
        "active_training": None,
        "current_best_method": "vanilla KD (Task 4 control)",
        "current_best_validation_mean": 0.9512,
        "dkd_validation_mean": summaries["DKD"].get("validation_mean"),
        "dist_validation_mean": summaries["DIST"].get("validation_mean"),
        "official_test": "LOCKED_NOT_RUN",
        "next_step": "Resume deep research with a separately reviewed one-seed experiment.",
    }
    checkpoint_state(active_training=None, final_method="vanilla KD (Task 4 control)",
                     final_validation_mean=0.9512, final_validation_std=0.00151,
                     final_summary="results/task4/task4_b3_final_t2_lam09_rerun1_summary.json",
                     execution_status=evidence,
                     next_experiment_after_completion=evidence["next_step"])
    print(json.dumps(evidence, indent=2), flush=True)

if __name__ == "__main__":
    main()
