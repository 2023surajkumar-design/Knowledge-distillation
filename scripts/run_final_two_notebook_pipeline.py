#!/usr/bin/env python3
"""Validation-only backend for final-two.ipynb.

It verifies the frozen Task 1--4 lineage and then runs an isolated, fresh QTRD
screen.  It intentionally does not re-run earlier completed experiments and
does not construct or access a test loader.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "research_state_checkpoint.json"
TRAINER = ROOT / "scripts/train_student_vanilla_kd.py"
CONFIG = ROOT / "configs/kd/resnet18_ternary_qtrd_finaltwo_screen.yaml"
SMOKE_NAME = "task11_qtrd_finaltwo_smoke_r1"
SCREEN_NAME = "task11_qtrd_finaltwo_screen_t2_lam09_aux1_r1"


def summary(run_name: str) -> Path:
    return ROOT / "results/task4" / f"{run_name}_summary.json"


def load_summary(run_name: str) -> dict:
    path = summary(run_name)
    payload = json.loads(path.read_text())
    if payload.get("test_evaluation") != "not_run":
        raise RuntimeError(f"{run_name} violated the validation-only firewall")
    if not payload.get("all_conv_and_fc_ternary"):
        raise RuntimeError(f"{run_name} is not an all-conv/all-FC ternary run")
    return payload


def update_state(**changes: object) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state.update(
        {
            "timestamp": datetime.now().astimezone().isoformat(),
            "notebook": "final-two.ipynb",
            "run_mode": "FINAL_TWO_VALIDATION_ONLY_QTRD_SCREEN",
            "test_evaluation": "LOCKED_NOT_RUN",
            **changes,
        }
    )
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def invoke(*extra: str) -> None:
    subprocess.run(
        [sys.executable, str(TRAINER), "--kd-mode", "qtrd", "--config", str(CONFIG), *extra],
        cwd=ROOT,
        check=True,
    )


def require_prior_lineage() -> None:
    required = (
        ROOT / "resnet34_cifar10_fp32_best.pth",
        ROOT / "results/best_models/task2_b1_fp32_resnet18_best.pth",
        ROOT / "results/best_models/task3_b2_default_best.pth",
        ROOT / "results/task3_b2_default_summary.json",
        ROOT / "results/task4/task4_b3_final_t2_lam09_aggregate_summary.json",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen prior artifacts: {missing}")
    task4 = json.loads((ROOT / "results/task4/task4_b3_final_t2_lam09_aggregate_summary.json").read_text())
    if task4.get("test_evaluation") != "not_run":
        raise RuntimeError("Task 4 aggregate must remain validation-only")


def main() -> None:
    require_prior_lineage()
    if not summary(SMOKE_NAME).exists():
        update_state(active_training={"run_name": SMOKE_NAME, "mode": "qtrd", "stage": "smoke"})
        invoke("--run-name", SMOKE_NAME, "--condition", "T6-QTRD-finaltwo-smoke", "--smoke", "--no-gradient-diagnostics")
    smoke = load_summary(SMOKE_NAME)
    if not summary(SCREEN_NAME).exists():
        update_state(active_training={"run_name": SCREEN_NAME, "mode": "qtrd", "stage": "one_seed_screen"})
        invoke()
    screen = load_summary(SCREEN_NAME)
    result = {
        "validation_mean": screen["validation_mean"],
        "validation_best": screen["validation_best"],
        "summary": str(summary(SCREEN_NAME).relative_to(ROOT)),
        "status": "one_seed_provisional",
        "smoke_summary": str(summary(SMOKE_NAME).relative_to(ROOT)),
    }
    update_state(
        active_training=None,
        final_two_qtrd_screen=result,
        final_two_decision="Task-4 vanilla KD remains the final control unless a separately authorized independent confirmation promotes a challenger.",
        next_experiment_after_completion="Review QTRD validation evidence; keep the official test set locked.",
    )
    print(json.dumps({"status": "COMPLETE", "qtrd": result, "smoke_validation": smoke["validation_mean"], "official_test": "not_run"}, indent=2))


if __name__ == "__main__":
    main()
