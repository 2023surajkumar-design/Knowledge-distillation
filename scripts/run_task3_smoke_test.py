#!/usr/bin/env python3
"""Fast real-data Task 3 invariants: no test-set access and no scientific result."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_student_ternary import build_ternary_student
from src.data import get_train_val_loaders
from src.evaluation.verify_ternary import verify_model
from src.quant.ternary import QuantConfig, TernaryConv2d, TernaryLinear, count_ternary_layers, parameter_group_audit, ternary_parameter_groups
from src.training.utils import get_device, set_seed


def main() -> None:
    set_seed(42)
    device = get_device(require_cuda=True)
    checkpoint = "results/best_models/task2_b1_fp32_resnet18_best.pth"
    cfg = QuantConfig().validate()
    model, warmed = build_ternary_student(cfg, checkpoint, False, device)
    if not warmed:
        raise AssertionError("Smoke test must prove the canonical Task 2 warm-start works.")
    conv, linear = count_ternary_layers(model)
    if (conv, linear) != (20, 1) or not isinstance(model.conv1, TernaryConv2d) or not isinstance(model.fc, TernaryLinear):
        raise AssertionError(f"Official B2 layer coverage failed: {(conv, linear)}")
    groups = ternary_parameter_groups(model, 1e-4)
    audit = parameter_group_audit(model, groups)
    if not audit["only_conv_linear_latents_decayed"] or audit["bn_bias_and_scale_tensors_decayed"]:
        raise AssertionError(audit)
    train_loader, _, _, _ = get_train_val_loaders("./data", batch_size=16, num_workers=0, loader_seed=42)
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer = torch.optim.SGD(groups, lr=1e-3, momentum=0.9)
    before = model.conv1.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(inputs), labels)
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite one-batch QAT loss.")
    loss.backward()
    if model.conv1.weight.grad is None or not torch.isfinite(model.conv1.weight.grad).all():
        raise AssertionError("No finite STE gradient reached the latent stem weight.")
    optimizer.step()
    if torch.equal(before, model.conv1.weight.detach()):
        raise AssertionError("Latent stem weight did not update through STE.")
    info = verify_model(model, require_all_weight_layers=True, verbose=False)
    if not info["all_ternary"] or not info["all_weight_layers_ternary"]:
        raise AssertionError(info)
    print("TASK 3 SMOKE PASSED: warm start, STE update, parameter groups, and strict deployed ternary coverage verified.")


if __name__ == "__main__":
    main()
