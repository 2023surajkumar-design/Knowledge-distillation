#!/usr/bin/env python3
"""Bounded numerical Task 3 HPO, persisted in SQLite and never using test data.

This is deliberately restricted to learning-rate, decay, warm-up, TWN factor,
and STE clip. It neither changes architecture/data/split nor relaxes the all
conv+FC strict ternary constraint. Hyperband receives final 5-epoch fidelity
reports; retain its database rather than cherry-picking unlogged trial outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_student_ternary import assert_fresh_outputs, output_paths, train_one_seed
from src.quant.ternary import QuantConfig
from src.training.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only numerical HPO for frozen Task 3 B2 policy.")
    parser.add_argument("--trials", type=int, default=0, help="Zero creates/opens the persisted study without training.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage", default="sqlite:///hpo/task3_optuna.sqlite3")
    args = parser.parse_args()
    if args.trials < 0 or args.epochs < 1:
        raise ValueError("trials must be non-negative and epochs positive.")
    study = optuna.create_study(
        study_name="task3_b2_numerical_screen", storage=args.storage, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=args.epochs, reduction_factor=3),
    )
    print(f"Study {study.study_name}: {args.storage}; completed trials={len(study.trials)}; test locked.")
    if args.trials == 0:
        return
    base = yaml.safe_load((REPO_ROOT / "configs/ternary/resnet18_ternary_noKD.yaml").read_text())
    device = get_device(require_cuda=True)

    def objective(trial: optuna.Trial) -> float:
        config = dict(base)
        config["quant"] = dict(base["quant"])
        config.update({
            "condition": "T3-HPO-screen", "run_name": f"task3_hpo_trial_{trial.number:03d}",
            "epochs": args.epochs, "seeds": [args.seed], "smoke": False,
            "lr": trial.suggest_float("lr", 0.02, 0.15, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
            "warmup_epochs": trial.suggest_int("warmup_epochs", 1, min(8, args.epochs)),
        })
        config["quant"].update({
            "twn_factor": trial.suggest_float("twn_factor", 0.5, 0.9),
            "clip": trial.suggest_float("clip", 0.5, 2.0),
        })
        namespace = argparse.Namespace(**config)
        quant = QuantConfig(**namespace.quant).validate()
        paths = output_paths(namespace.run_name)
        assert_fresh_outputs(paths, namespace.seeds)
        result = train_one_seed(args.seed, namespace, quant, device, paths, REPO_ROOT / "configs/ternary/resnet18_ternary_noKD.yaml")
        trial.report(result["best_validation_accuracy"], step=args.epochs)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return result["best_validation_accuracy"]

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    print(f"Best validation-only trial: {study.best_trial.number} -> {study.best_value:.4f}; do not replace B2-default without fresh-seed confirmation.")


if __name__ == "__main__":
    main()
