#!/usr/bin/env python3
"""Executable backend for final.ipynb: sequential one-seed, validation-only screens."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "research_state_checkpoint.json"
STAGES = (
    ("qfd", "configs/kd/resnet18_ternary_qfd_screen.yaml", "task8_qfd_smoke_r1", "T6-QFD-smoke", "task8_qfd_screen_t2_lam09_aux1_r1"),
    ("at", "configs/kd/resnet18_ternary_at_screen.yaml", "task9_at_smoke_r1", "T7-AT-smoke", "task9_at_screen_t2_lam09_aux1_r1"),
    ("rkd", "configs/kd/resnet18_ternary_rkd_screen.yaml", "task10_rkd_smoke_r1", "T8-RKD-smoke", "task10_rkd_screen_t2_lam09_aux1_r1"),
    ("qtrd", "configs/kd/resnet18_ternary_qtrd_screen.yaml", "task11_qtrd_smoke_r1", "T6-QTRD-smoke", "task11_qtrd_screen_t2_lam09_aux1_r1"),
)

def summary(run: str) -> Path:
    return ROOT / "results/task4" / f"{run}_summary.json"

def update_state(**changes: object) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state.update({"timestamp": datetime.now().astimezone().isoformat(),
                  "notebook": "final.ipynb", "run_mode": "FINAL_NOTEBOOK_ONE_SEED_SCREENS",
                  "test_evaluation": "LOCKED_NOT_RUN", **changes})
    STATE.write_text(json.dumps(state, indent=2) + "\n")

def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)

def main() -> None:
    trainer = ROOT / "scripts/train_student_vanilla_kd.py"
    results = {}
    for mode, config_rel, smoke_name, smoke_condition, screen_name in STAGES:
        config = ROOT / config_rel
        if not summary(smoke_name).exists():
            update_state(active_training={"run_name": smoke_name, "mode": mode, "stage": "smoke"})
            run([sys.executable, str(trainer), "--kd-mode", mode, "--config", str(config),
                 "--run-name", smoke_name, "--condition", smoke_condition, "--smoke", "--no-gradient-diagnostics"])
        smoke = json.loads(summary(smoke_name).read_text())
        if smoke["test_evaluation"] != "not_run" or not smoke["all_conv_and_fc_ternary"]:
            raise RuntimeError(f"{mode} smoke failed safety checks")
        if not summary(screen_name).exists():
            update_state(active_training={"run_name": screen_name, "mode": mode, "stage": "one_seed_screen"})
            run([sys.executable, str(trainer), "--kd-mode", mode, "--config", str(config)])
        result = json.loads(summary(screen_name).read_text())
        if result["test_evaluation"] != "not_run" or not result["all_conv_and_fc_ternary"]:
            raise RuntimeError(f"{mode} screen failed safety checks")
        results[mode] = {"validation_mean": result["validation_mean"], "validation_best": result["validation_best"],
                         "summary": str(summary(screen_name).relative_to(ROOT)), "status": "one_seed_provisional"}
    winner = max(results, key=lambda key: results[key]["validation_mean"])
    update_state(active_training=None, final_notebook_screens=results,
                 final_notebook_best_screen=winner,
                 final_notebook_decision="Task-4 vanilla KD remains final control unless a separately authorized confirmation is run.",
                 next_experiment_after_completion="Review one-seed QFD/AT/RKD evidence; do not promote a screen to final without independent confirmation.")
    print(json.dumps({"status": "COMPLETE", "screens": results, "best_screen": winner,
                      "official_test": "not_run"}, indent=2))

if __name__ == "__main__":
    main()
