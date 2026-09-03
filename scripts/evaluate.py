#!/usr/bin/env python3
"""Final-only official CIFAR-10 evaluation; it is deliberately unavailable to training code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import get_test_loader, get_train_val_loaders
from src.models.resnet_cifar import resnet18_cifar, resnet34_cifar
from src.training.utils import evaluate, get_device, save_json

ARCHITECTURES = {"resnet18": resnet18_cifar, "resnet34": resnet34_cifar}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final, firewalled CIFAR-10 evaluation")
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--arch", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", default="results/task2_test_evaluation.json")
    parser.add_argument("--final-evaluation", action="store_true",
                        help="required: unlocks the official test set after all decisions are frozen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.final_evaluation:
        raise SystemExit("Refusing test access: pass --final-evaluation only for frozen final reporting.")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an immutable test report: {output}")
    device = get_device(require_cuda=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # This is the sole explicit test-loader unlock in the project source.
    test_loader = get_test_loader(args.data_root, args.batch_size, args.num_workers, final_evaluation=True)
    _, validation_loader, _, _ = get_train_val_loaders(
        args.data_root, args.batch_size, args.num_workers, loader_seed=42,
    )
    rows = []
    for checkpoint_path_str in args.checkpoint:
        checkpoint_path = Path(checkpoint_path_str)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected_arch = "ResNet18-CIFAR" if args.arch == "resnet18" else "ResNet34-CIFAR"
        if checkpoint.get("arch") != expected_arch:
            raise ValueError(f"{checkpoint_path} has arch={checkpoint.get('arch')!r}, expected {expected_arch!r}.")
        model = ARCHITECTURES[args.arch](num_classes=10).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        rows.append({
            "checkpoint": str(checkpoint_path), "seed": checkpoint.get("seed"),
            "selected_epoch": checkpoint.get("epoch"),
            "checkpoint_best_validation_accuracy": checkpoint.get("best_validation_accuracy"),
            "recomputed_validation_accuracy": validation_accuracy, "validation_loss": validation_loss,
            "test_accuracy": test_accuracy, "test_loss": test_loss,
        })
        print(f"{checkpoint_path.name} | seed {checkpoint.get('seed')} | "
              f"validation {100*validation_accuracy:.2f}% | test {100*test_accuracy:.2f}%")
    accuracies = np.asarray([row["test_accuracy"] for row in rows])
    save_json(output, {
        "purpose": "final_reporting_only", "architecture": args.arch,
        "per_checkpoint": rows, "test_accuracy_mean": float(accuracies.mean()),
        "test_accuracy_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
        "note": "Independent checkpoint evaluations; no ensemble and no test-based model selection.",
    })
    print(f"Final independent-checkpoint test accuracy: {100*accuracies.mean():.2f}% ± "
          f"{100*(accuracies.std(ddof=1) if len(accuracies) > 1 else 0.0):.2f}%")
    print(f"Saved immutable report: {output}")


if __name__ == "__main__":
    main()
