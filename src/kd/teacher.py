"""Frozen Task 1 teacher loading and invariants for Task 4."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from src.models.resnet_cifar import resnet34_cifar


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_teacher_frozen(teacher: nn.Module, assert_no_gradients: bool = False) -> None:
    if teacher.training:
        raise AssertionError("Teacher must remain in eval mode.")
    trainable = [name for name, parameter in teacher.named_parameters() if parameter.requires_grad]
    if trainable:
        raise AssertionError(f"Teacher has trainable parameters: {trainable[:3]}")
    if assert_no_gradients:
        gradients = [name for name, parameter in teacher.named_parameters() if parameter.grad is not None]
        if gradients:
            raise AssertionError(f"Teacher received gradients: {gradients[:3]}")


def load_frozen_teacher(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("arch") != "ResNet34-CIFAR" or checkpoint.get("num_classes") != 10:
        raise ValueError("Task 4 requires the frozen 10-class Task 1 ResNet34 teacher.")
    teacher = resnet34_cifar(num_classes=10)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    assert_teacher_frozen(teacher)
    metadata = {
        "path": str(path), "sha256": sha256_file(path), "arch": checkpoint["arch"],
        "parameter_count": sum(parameter.numel() for parameter in teacher.parameters()),
        "device": str(device), "mode": "eval", "requires_grad_any": False,
        "selection_metric": checkpoint.get("selection_metric"),
        "best_validation_accuracy": checkpoint.get("best_validation_accuracy", checkpoint.get("best_accuracy")),
    }
    return teacher, checkpoint, metadata
