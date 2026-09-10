"""Numerically stable vanilla logit-KD loss for Task 4.

This module intentionally contains only conventional logit distillation:
``(1-lambda) CE + lambda T^2 KL(p_teacher || p_student)``.  It contains no
feature, relation, contrastive, or adaptive KD components.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VanillaKDLoss:
    total: torch.Tensor
    ce: torch.Tensor
    kd: torch.Tensor
    weighted_ce: torch.Tensor
    weighted_kd: torch.Tensor


def decoupled_kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor,
    temperature: float, kd_lambda: float, ce_criterion: nn.Module,
    alpha: float = 1.0, beta: float = 8.0,
) -> VanillaKDLoss:
    """DKD: target-class and non-target-class KL terms, validation-safe.

    The outer CE/KD mixture keeps the Task-4 control's objective convention;
    ``alpha`` and ``beta`` only decompose the KD component.  No test data is
    involved and teacher logits are expected to be detached by the caller.
    """
    if temperature <= 0 or not 0 <= kd_lambda <= 1 or alpha < 0 or beta < 0:
        raise ValueError("Invalid DKD hyperparameters.")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and student logits must have identical shapes.")
    ce = ce_criterion(student_logits, targets)
    t = float(temperature)
    student_prob = F.softmax(student_logits / t, dim=1)
    teacher_prob = F.softmax(teacher_logits / t, dim=1)
    target_mask = F.one_hot(targets, num_classes=student_logits.size(1)).bool()
    s_target = student_prob[target_mask]; t_target = teacher_prob[target_mask]
    two_s = torch.stack((s_target, 1.0 - s_target), dim=1).clamp_min(1e-12)
    two_t = torch.stack((t_target, 1.0 - t_target), dim=1).clamp_min(1e-12)
    tckd = F.kl_div(two_s.log(), two_t, reduction="batchmean") * t**2
    # Materialise the K-1 non-target logits: masking to -inf would expose an
    # undefined 0 * -inf term in KL's target-class slot.
    non_target_mask = ~target_mask
    student_non_target = (student_logits[non_target_mask].view(student_logits.size(0), -1) / t)
    teacher_non_target = (teacher_logits[non_target_mask].view(teacher_logits.size(0), -1) / t)
    nckd = F.kl_div(F.log_softmax(student_non_target, dim=1), F.softmax(teacher_non_target, dim=1), reduction="batchmean") * t**2
    kd = alpha * tckd + beta * nckd
    weighted_ce, weighted_kd = (1.0-kd_lambda)*ce, kd_lambda*kd
    return VanillaKDLoss(total=weighted_ce+weighted_kd, ce=ce, kd=kd,
                         weighted_ce=weighted_ce, weighted_kd=weighted_kd)


def vanilla_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    kd_lambda: float,
    ce_criterion: nn.Module,
) -> VanillaKDLoss:
    """Return the canonical Hinton loss with ``KL(teacher || student)``.

    PyTorch's ``kl_div`` expects the student's log-probabilities as its input and
    teacher probabilities as its target.  The teacher logits must be detached or
    produced under ``torch.no_grad`` by the caller.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 <= kd_lambda <= 1:
        raise ValueError("kd_lambda must be in [0, 1].")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and student logits must have identical shapes.")
    ce = ce_criterion(student_logits, targets)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probabilities = F.softmax(teacher_logits / temperature, dim=1)
    kd = F.kl_div(log_student, teacher_probabilities, reduction="batchmean") * temperature**2
    weighted_ce = (1.0 - kd_lambda) * ce
    weighted_kd = kd_lambda * kd
    return VanillaKDLoss(
        total=weighted_ce + weighted_kd,
        ce=ce,
        kd=kd,
        weighted_ce=weighted_ce,
        weighted_kd=weighted_kd,
    )
