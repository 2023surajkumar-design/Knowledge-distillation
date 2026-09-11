#!/usr/bin/env python3
"""One-time official CIFAR-10 evaluation of frozen ATDL checkpoints.

This script exists outside training. It refuses to run without an explicit final
unlock and refuses to overwrite a prior report, preventing test-driven tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_test_loader, get_train_val_loaders
from src.evaluation.verify_ternary import verify_model
from src.models.resnet_cifar import resnet18_cifar, resnet34_cifar
from src.quant.ternary import QuantConfig, convert_to_ternary
from src.training.utils import get_device, save_json

OUTPUT = ROOT / "results" / "final_official_test_evaluation.json"

FROZEN_GROUPS = {
    "FP32 ResNet34 teacher": [ROOT / "resnet34_cifar10_fp32_best.pth"],
    "FP32 ResNet18 baseline": [
        ROOT / f"experiments/checkpoints/task2_b1_fp32_resnet18/resnet18_fp32_seed{seed}.pth" for seed in (42, 43, 44)
    ],
    "Ternary ResNet18 (no KD)": [
        ROOT / f"experiments/checkpoints/task3_b2_default/resnet18_ternary_seed{seed}.pth" for seed in (42, 43, 44)
    ],
    "Vanilla ternary KD (final control)": [
        ROOT / "experiments/task4/checkpoints/task4_b3_final_t2_lam09_rerun1/resnet18_ternary_kd_seed42.pth",
        ROOT / "experiments/task4/checkpoints/task4_b3_final_t2_lam09_remaining_rerun1/resnet18_ternary_kd_seed43.pth",
        ROOT / "experiments/task4/checkpoints/task4_b3_final_t2_lam09_remaining_rerun1/resnet18_ternary_kd_seed44.pth",
    ],
    "QTRD exploratory screen": [
        ROOT / "experiments/task4/checkpoints/task11_qtrd_finaltwo_screen_t2_lam09_aux1_r1/resnet18_ternary_kd_seed42.pth"
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final-only official CIFAR-10 checkpoint evaluation")
    parser.add_argument("--final-evaluation", action="store_true", help="explicitly unlocks official test data")
    return parser.parse_args()


def build_model(checkpoint: dict, device: torch.device) -> tuple[nn.Module, dict | None]:
    arch = checkpoint.get("arch")
    if arch == "ResNet34-CIFAR":
        model = resnet34_cifar(num_classes=10).to(device)
        ternary = None
    elif arch == "ResNet18-CIFAR":
        model = resnet18_cifar(num_classes=10).to(device)
        ternary = None
    elif arch == "ResNet18-CIFAR-ternary":
        model = resnet18_cifar(num_classes=10).to(device)
        quant = checkpoint.get("quant_config")
        if not quant:
            raise ValueError("ternary checkpoint is missing quant_config")
        convert_to_ternary(model, QuantConfig(**quant), keep_first_last_fp32=bool(checkpoint.get("keep_first_last_fp32", False)))
        ternary = "pending"
    else:
        raise ValueError(f"unsupported checkpoint architecture: {arch!r}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if ternary == "pending":
        ternary = verify_model(model, require_all_weight_layers=not checkpoint.get("keep_first_last_fp32", False), verbose=False)
    model.eval()
    return model, ternary


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> tuple[float, float, list[list[int]]]:
    total_loss = total_correct = total = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, labels).item()) * labels.numel()
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total += labels.numel()
        confusion.index_put_((labels.cpu(), predictions.cpu()), torch.ones_like(labels.cpu(), dtype=torch.int64), accumulate=True)
    return total_loss / total, total_correct / total, confusion.tolist()


def main() -> None:
    args = parse_args()
    if not args.final_evaluation:
        raise SystemExit("Refusing test access: pass --final-evaluation only after model selection is frozen.")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite immutable official report: {OUTPUT}")
    missing = [str(path.relative_to(ROOT)) for paths in FROZEN_GROUPS.values() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen checkpoint(s): {missing}")

    device = get_device(require_cuda=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loader = get_test_loader(ROOT / "data", batch_size=256, num_workers=4, final_evaluation=True)
    _, validation_loader, _, _ = get_train_val_loaders(ROOT / "data", batch_size=256, num_workers=4, loader_seed=42)
    groups: dict[str, dict] = {}
    for name, checkpoints in FROZEN_GROUPS.items():
        rows = []
        for path in checkpoints:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model, ternary = build_model(checkpoint, device)
            validation_loss, validation_accuracy, _ = evaluate(model, validation_loader, device, criterion)
            test_loss, test_accuracy, confusion = evaluate(model, test_loader, device, criterion)
            rows.append({
                "checkpoint": str(path.relative_to(ROOT)), "seed": checkpoint.get("seed"), "architecture": checkpoint.get("arch"),
                "selected_epoch": checkpoint.get("epoch"), "checkpoint_best_validation_accuracy": checkpoint.get("best_validation_accuracy"),
                "recomputed_validation_accuracy": validation_accuracy, "validation_loss": validation_loss,
                "test_accuracy": test_accuracy, "test_loss": test_loss, "test_confusion_matrix": confusion,
                "ternary_verification": ternary,
            })
            print(f"{name} | {path.name} | val={100*validation_accuracy:.2f}% | test={100*test_accuracy:.2f}%")
            del model
            torch.cuda.empty_cache()
        values = np.asarray([row["test_accuracy"] for row in rows], dtype=float)
        groups[name] = {
            "n_checkpoints": len(rows), "test_accuracy_mean": float(values.mean()),
            "test_accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "per_checkpoint": rows,
        }

    report = {
        "purpose": "official_final_reporting_only", "generated": datetime.now().astimezone().isoformat(),
        "dataset": "CIFAR-10 official 10,000-example test set", "selection_policy": "All models/checkpoints were frozen before first test access; no test-based model selection or tuning occurred.",
        "final_control": "Vanilla ternary KD (Task 4), selected from its three-seed validation aggregate before test access.",
        "groups": groups,
    }
    save_json(OUTPUT, report)
    print(f"Saved immutable official report: {OUTPUT}")


if __name__ == "__main__":
    main()
