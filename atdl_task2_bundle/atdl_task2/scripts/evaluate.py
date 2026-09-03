#!/usr/bin/env python3
"""
scripts/evaluate.py  —  FINAL evaluation on the official CIFAR-10 test set.

TEST-SET FIREWALL (C9 / C11 / Phase 42): this is the ONLY script permitted to
read the test set, and only when invoked with the explicit --final-evaluation
flag. Do not import or call it during training, HPO, Optuna, AutoResearch, or
ablation search. Use it once, at the end, for the report.

Supports one or more checkpoints (e.g. the three Task 2 seeds) and reports
per-checkpoint test accuracy plus mean ± std, alongside the validation accuracy
each checkpoint was selected on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data import get_test_loader, get_train_val_loaders  # noqa: E402
from src.models.resnet_cifar import resnet18_cifar, resnet34_cifar  # noqa: E402
from src.training.utils import evaluate, get_device  # noqa: E402


ARCHS = {"resnet18": resnet18_cifar, "resnet34": resnet34_cifar}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final test-set evaluation (firewalled)")
    p.add_argument("--checkpoint", nargs="+", required=True,
                   help="one or more .pth checkpoints to evaluate")
    p.add_argument("--arch", choices=list(ARCHS), default="resnet18")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default="results/test_evaluation.json")
    p.add_argument("--final-evaluation", action="store_true", default=False,
                   help="REQUIRED. Unlocks the official test set for final reporting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.final_evaluation:
        raise SystemExit(
            "Refusing to run: the test set is locked (C9/C11). "
            "Pass --final-evaluation only when producing the final report."
        )

    device = get_device()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Unlock the test loader exactly once, here.
    test_loader = get_test_loader(
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, final_evaluation=True)
    # Validation loader (for reporting the selection metric alongside test).
    _, val_loader, _, _ = get_train_val_loaders(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)

    rows, test_accs = [], []
    for ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ARCHS[args.arch](num_classes=10).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_accs.append(test_acc)
        rows.append({
            "checkpoint": ckpt_path, "seed": ckpt.get("seed"),
            "selected_epoch": ckpt.get("epoch"),
            "val_acc": val_acc, "val_loss": val_loss,
            "test_acc": test_acc, "test_loss": test_loss,
        })
        print(f"{os.path.basename(ckpt_path):40s} | seed {ckpt.get('seed')} | "
              f"val {100*val_acc:.2f}% | test {100*test_acc:.2f}%")

    test_accs = np.array(test_accs)
    summary = {
        "arch": args.arch, "per_checkpoint": rows,
        "test_acc_mean": float(test_accs.mean()),
        "test_acc_std": float(test_accs.std()),
        "test_acc_best": float(test_accs.max()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTEST accuracy: {100*test_accs.mean():.2f}% ± {100*test_accs.std():.2f} "
          f"(best {100*test_accs.max():.2f}%) over {len(rows)} checkpoint(s)")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
