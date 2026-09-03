#!/usr/bin/env python3
"""Staged, validation-only Task 4 temperature/lambda search.

Every subprocess uses the production trainer; the controller never accesses a
test loader and never overwrites an existing experimental result.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "scripts" / "train_student_vanilla_kd.py"
ROOT = REPO_ROOT / "experiments" / "task4" / "t_lambda"
TEMPERATURES = (1.0, 2.0, 4.0, 8.0, 16.0)
LAMBDAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_name(stage: str, temperature: float, kd_lambda: float) -> str:
    return f"task4_tlambda_{stage}_t{label(temperature)}_l{label(kd_lambda)}"


def summary_path(name: str) -> Path:
    return REPO_ROOT / "results" / "task4" / f"{name}_summary.json"


def execute(stage: str, temperature: float, kd_lambda: float, epochs: int, seeds: list[int]) -> dict:
    name = run_name(stage, temperature, kd_lambda)
    summary = summary_path(name)
    log = ROOT / stage / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not summary.exists():
        command = [
            sys.executable, str(TRAINER), "--run-name", name, "--condition", f"t_lambda_{stage}",
            "--temperature", str(temperature), "--kd-lambda", str(kd_lambda), "--epochs", str(epochs),
            "--seeds", *map(str, seeds), "--no-gradient-diagnostics",
        ]
        print(f"[{stage}] T={temperature:g}, lambda={kd_lambda:g}, epochs={epochs}, seeds={seeds}", flush=True)
        with log.open("w") as handle:
            completed = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Screening run failed ({name}); inspect {log}")
    payload = json.loads(summary.read_text())
    return {
        "run_name": name, "temperature": temperature, "kd_lambda": kd_lambda, "epochs": epochs,
        "seeds": seeds, "validation_mean": payload["validation_mean"], "validation_std": payload["validation_std"],
        "validation_best": payload["validation_best"], "summary": str(summary), "log": str(log),
        "test_evaluation": payload["test_evaluation"],
    }


def load_ranking(stage: str) -> list[dict]:
    path = ROOT / stage / "ranking.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run the preceding stage first: {path}")
    return json.loads(path.read_text())["ranking"]


def save_ranking(stage: str, rows: list[dict], selection_note: str) -> None:
    ranked = sorted(rows, key=lambda row: (row["validation_mean"], row["validation_best"]), reverse=True)
    payload = {
        "task": "T4", "stage": stage, "selection_metric": "validation accuracy only",
        "test_evaluation": "not_run", "selection_note": selection_note, "ranking": ranked,
    }
    path = ROOT / stage / "ranking.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[{stage}] wrote ranking: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 staged T/lambda screen (validation only).")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "integrity-control"], required=True)
    args = parser.parse_args()
    if args.stage == "stage1":
        rows = [execute("stage1", t, lam, epochs=6, seeds=[42]) for t, lam in itertools.product(TEMPERATURES, LAMBDAS)]
        save_ranking("stage1", rows, "All 25 candidates receive six epochs; retain top eight by validation accuracy.")
    elif args.stage == "stage2":
        parents = load_ranking("stage1")[:8]
        rows = [execute("stage2", row["temperature"], row["kd_lambda"], epochs=30, seeds=[42]) for row in parents]
        save_ranking("stage2", rows, "Stage-1 top eight receive 30 epochs; retain top three by validation accuracy.")
    elif args.stage == "stage3":
        parents = load_ranking("stage2")[:3]
        rows = [execute("stage3", row["temperature"], row["kd_lambda"], epochs=60, seeds=[42, 43]) for row in parents]
        save_ranking("stage3", rows, "Stage-2 top three receive 60 epochs on two fixed seeds; select by mean validation accuracy.")
    else:
        row = execute("integrity-control", temperature=4.0, kd_lambda=0.0, epochs=30, seeds=[42])
        save_ranking("integrity-control", [row], "KD-off integrity control; expected to match the no-KD objective, not used to select a KD temperature.")


if __name__ == "__main__":
    main()
