#!/usr/bin/env python3
"""Reproducible Optuna TPE/Hyperband refinement for Task 4 (validation only)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "experiments" / "task4" / "optimization"
TRAINER = REPO_ROOT / "scripts" / "train_student_vanilla_kd.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 Optuna refinement; never touches the test set.")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.trials < 1 or args.epochs < 2:
        raise ValueError("Use at least one trial and two epochs.")
    ROOT.mkdir(parents=True, exist_ok=True)
    study_path = ROOT / "task4_optuna.sqlite3"
    study = optuna.create_study(
        study_name="task4_vanilla_kd_validation_only",
        storage=f"sqlite:///{study_path}", load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026, multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(min_resource=3, max_resource=args.epochs, reduction_factor=3),
    )

    def objective(trial: optuna.Trial) -> float:
        temperature = trial.suggest_float("temperature", 1.0, 16.0, log=True)
        kd_lambda = trial.suggest_float("kd_lambda", 0.1, 0.9)
        lr = trial.suggest_float("lr", 0.05, 0.15, log=True)
        wd = trial.suggest_float("weight_decay", 3e-5, 3e-4, log=True)
        name = f"task4_optuna_trial{trial.number:03d}"
        summary_path = REPO_ROOT / "results" / "task4" / f"{name}_summary.json"
        log_path = ROOT / f"{name}.log"
        if not summary_path.exists():
            command = [
                sys.executable, str(TRAINER), "--run-name", name, "--condition", "optuna-validation-only",
                "--temperature", str(temperature), "--kd-lambda", str(kd_lambda), "--lr", str(lr),
                "--weight-decay", str(wd), "--epochs", str(args.epochs), "--seeds", "42",
                "--no-gradient-diagnostics",
            ]
            with log_path.open("w") as handle:
                completed = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Trial {trial.number} failed; inspect {log_path}")
        summary = json.loads(summary_path.read_text())
        history_path = Path(summary["per_seed"][0]["history"])
        history = json.loads(history_path.read_text())["history"]["val_acc"]
        for epoch, value in enumerate(history, 1):
            trial.report(value, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Validation-pruned after completed short trial at epoch {epoch}.")
        trial.set_user_attr("summary", str(summary_path))
        trial.set_user_attr("test_evaluation", "not_run")
        return float(summary["validation_mean"])

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    payload = {
        "task": "T4", "study": study.study_name, "storage": str(study_path),
        "sampler": "TPESampler(seed=2026,multivariate=True)", "pruner": "HyperbandPruner",
        "completed_trials": len(completed), "requested_new_trials": args.trials, "epochs_per_trial": args.epochs,
        "best_validation_trial": None if not completed else {"number": study.best_trial.number, "value": study.best_value, "params": study.best_params},
        "test_evaluation": "not_run",
    }
    out = ROOT / "task4_optuna_summary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote validation-only Optuna study summary: {out}")


if __name__ == "__main__":
    main()
