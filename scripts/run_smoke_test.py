#!/usr/bin/env python3
"""Fast real-CIFAR infrastructure test; produces no scientific Task 2 result."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import get_test_loader, get_train_val_loaders
from src.models.resnet_cifar import count_parameters, resnet18_cifar, resnet34_cifar
from src.training.utils import (
    evaluate, get_device, make_warmup_cosine, save_checkpoint, save_training_curves, set_seed, train_one_epoch,
)


def evenly_spaced_subset(dataset, size: int) -> Subset:
    indices = torch.linspace(0, len(dataset) - 1, steps=size).round().to(torch.int64).tolist()
    return Subset(dataset, indices)


def main() -> None:
    set_seed(42)
    device = get_device(require_cuda=True)
    student = resnet18_cifar(10).to(device)
    teacher = resnet34_cifar(10).to(device)
    assert count_parameters(student) == 11_173_962
    assert count_parameters(teacher) == 21_282_122
    teacher_checkpoint = torch.load(REPO_ROOT / "resnet34_cifar10_fp32_final.pth", map_location=device, weights_only=False)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        assert student(torch.zeros(4, 3, 32, 32, device=device)).shape == (4, 10)
        assert teacher(torch.zeros(4, 3, 32, 32, device=device)).shape == (4, 10)
    print("Model counts, shapes, and Task 1 teacher compatibility: PASS")

    train_loader, validation_loader, train_indices, validation_indices = get_train_val_loaders(
        batch_size=128, num_workers=2, download=False,
    )
    assert len(train_indices) == 45_000 and len(validation_indices) == 5_000
    assert set(train_indices).isdisjoint(validation_indices)
    # Spacing avoids the class-0-only first contiguous chunk of a class-grouped split.
    small_train = DataLoader(evenly_spaced_subset(train_loader.dataset, 512), batch_size=128, shuffle=True)
    small_validation = DataLoader(evenly_spaced_subset(validation_loader.dataset, 256), batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = make_warmup_cosine(optimizer, warmup_epochs=0, total_epochs=1)
    train_loss, train_accuracy = train_one_epoch(student, small_train, criterion, optimizer, device)
    scheduler.step()
    validation_loss, validation_accuracy = evaluate(student, small_validation, criterion, device)
    if not all(torch.isfinite(torch.tensor(value)) for value in [train_loss, train_accuracy, validation_loss, validation_accuracy]):
        raise AssertionError("Smoke metrics are non-finite.")
    print(f"Real-CIFAR subset: train {train_loss:.3f}/{100*train_accuracy:.1f}% | "
          f"validation {validation_loss:.3f}/{100*validation_accuracy:.1f}%")

    with tempfile.TemporaryDirectory(prefix="atdl_task2_smoke_") as temporary_directory:
        temporary = Path(temporary_directory)
        checkpoint_path = temporary / "smoke.pth"
        save_checkpoint(
            checkpoint_path, student, optimizer, scheduler, 1, validation_accuracy,
            {"seed": 42, "arch": "ResNet18-CIFAR", "num_classes": 10, "split_manifest": {"smoke": True}, "runtime": {}},
        )
        reloaded = resnet18_cifar(10).to(device)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        reloaded.load_state_dict(saved["model_state_dict"], strict=True)
        reload_loss, reload_accuracy = evaluate(reloaded, small_validation, criterion, device)
        assert abs(reload_accuracy - validation_accuracy) < 1e-12 and abs(reload_loss - validation_loss) < 1e-12
        plots = save_training_curves(
            {"train_loss": [train_loss], "train_acc": [train_accuracy], "val_loss": [validation_loss],
             "val_acc": [validation_accuracy], "lr": [0.01], "gap": [train_accuracy-validation_accuracy]},
            temporary / "curves",
        )
        assert len(plots) == 4 and all(Path(plot).is_file() for plot in plots)
    print("Checkpoint round-trip and four plot exports: PASS")
    try:
        get_test_loader(download=False)
    except RuntimeError:
        print("Test firewall refusal without final permission: PASS")
    else:
        raise AssertionError("Test firewall failed.")
    print("SMOKE TEST PASSED — infrastructure only; no official result or test evaluation was produced.")


if __name__ == "__main__":
    main()
