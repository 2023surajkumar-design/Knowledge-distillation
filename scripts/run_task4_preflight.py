#!/usr/bin/env python3
"""Task 4 KD correctness and frozen-teacher preflight (validation/test safe)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_student_vanilla_kd import build_student
from src.data import get_train_val_loaders
from src.evaluation.verify_ternary import verify_model
from src.kd.losses import vanilla_kd_loss
from src.kd.teacher import assert_teacher_frozen, load_frozen_teacher
from src.quant.ternary import QuantConfig, TernaryConv2d, TernaryLinear
from src.training.utils import get_device, save_json, set_seed


def unit_tests() -> dict:
    torch.manual_seed(7)
    student = torch.tensor([[1.1, -0.4, 0.2], [0.3, 0.8, -0.7]], requires_grad=True)
    teacher = torch.tensor([[0.7, -0.1, 0.4], [-0.2, 1.2, -0.3]])
    targets = torch.tensor([0, 2])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce = criterion(student, targets)
    only_ce = vanilla_kd_loss(student, teacher, targets, 4.0, 0.0, criterion)
    only_kd = vanilla_kd_loss(student, teacher, targets, 4.0, 1.0, criterion)
    manual_kd = F.kl_div(F.log_softmax(student / 4.0, dim=1), F.softmax(teacher / 4.0, dim=1), reduction="batchmean") * 16.0
    same = vanilla_kd_loss(student, student.detach(), targets, 4.0, 1.0, criterion)
    reverse = F.kl_div(F.log_softmax(teacher / 4.0, dim=1), F.softmax(student.detach() / 4.0, dim=1), reduction="batchmean") * 16.0
    checks = {
        "lambda_zero_is_ce": bool(torch.allclose(only_ce.total, ce, atol=1e-7)),
        "lambda_one_is_kd": bool(torch.allclose(only_kd.total, manual_kd, atol=1e-7)),
        "manual_direction_matches": bool(torch.allclose(only_kd.kd, manual_kd, atol=1e-7)),
        "identical_logits_have_zero_kd": bool(abs(float(same.kd.detach())) < 1e-6),
        "forward_and_reverse_kl_not_conflated": bool(not torch.allclose(manual_kd, reverse, atol=1e-5)),
        "t_squared_applied": bool(torch.allclose(only_kd.kd / 16.0, F.kl_div(F.log_softmax(student / 4.0, dim=1), F.softmax(teacher / 4.0, dim=1), reduction="batchmean"), atol=1e-7)),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def real_batch_test() -> dict:
    set_seed(42)
    device = get_device(require_cuda=True)
    cfg = QuantConfig(threshold="twn", scale="derived", symmetric=True, scope="per_channel", t=0.05, twn_factor=0.7, ste="clipped", clip=1.0).validate()
    teacher, _, teacher_metadata = load_frozen_teacher(REPO_ROOT / "resnet34_cifar10_fp32_best.pth", device)
    train_loader, _, _, _ = get_train_val_loaders("./data", batch_size=16, num_workers=0, loader_seed=42)
    student = build_student(cfg, "results/best_models/task2_b1_fp32_resnet18_best.pth", device)
    student.train()
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        teacher_logits = teacher(inputs)
    student_logits = student(inputs)
    loss = vanilla_kd_loss(student_logits, teacher_logits, labels, temperature=4.0, kd_lambda=0.5, ce_criterion=nn.CrossEntropyLoss(label_smoothing=0.1))
    loss.total.backward()
    assert_teacher_frozen(teacher, assert_no_gradients=True)
    ternary = verify_model(student, require_all_weight_layers=True, verbose=False)
    latent_gradients = [module.weight.grad for module in student.modules() if isinstance(module, (TernaryConv2d, TernaryLinear))]
    result = {
        "teacher": teacher_metadata,
        "batch_size": int(labels.size(0)),
        "loss_total": float(loss.total.detach()), "loss_ce": float(loss.ce.detach()), "loss_kd": float(loss.kd.detach()),
        "student_received_gradients": bool(any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in latent_gradients)),
        "teacher_frozen_and_grad_free": True,
        "all_conv_and_fc_ternary": bool(ternary["all_ternary"]),
        "ternary_layer_count": int(ternary["num_ternary_layers"]), "sparsity": float(ternary["sparsity"]),
        "test_evaluation": "not_run",
    }
    if not result["student_received_gradients"] or not result["all_conv_and_fc_ternary"]:
        raise AssertionError(result)
    return result


def main() -> None:
    result = {"task": "T4", "unit_tests": unit_tests(), "real_batch": real_batch_test(), "test_loader_imported": False}
    output = REPO_ROOT / "experiments" / "task4" / "smoke" / "task4_preflight.json"
    save_json(output, result)
    print(f"Task 4 preflight passed; evidence written to {output}")


if __name__ == "__main__":
    main()
