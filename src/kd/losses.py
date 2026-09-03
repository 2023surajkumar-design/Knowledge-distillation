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
