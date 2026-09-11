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


def dist_loss(student_logits, teacher_logits, targets, temperature, kd_lambda, ce_criterion,
              inter_weight: float = 1.0, intra_weight: float = 1.0) -> VanillaKDLoss:
    """One-batch DIST screen: transfer inter/intra class probability relations."""
    if temperature <= 0 or not 0 <= kd_lambda <= 1:
        raise ValueError("Invalid DIST hyperparameters.")
    ce = ce_criterion(student_logits, targets)
    sp = F.softmax(student_logits / temperature, dim=1)
    tp = F.softmax(teacher_logits / temperature, dim=1)
    def correlation(a, b, dim):
        a = a - a.mean(dim=dim, keepdim=True); b = b - b.mean(dim=dim, keepdim=True)
        return (a*b).sum(dim=dim) / (a.square().sum(dim=dim).sqrt()*b.square().sum(dim=dim).sqrt() + 1e-12)
    inter = 1.0 - correlation(sp, tp, dim=1).mean()
    intra = 1.0 - correlation(sp, tp, dim=0).mean()
    kd = (inter_weight * inter + intra_weight * intra) * temperature**2
    weighted_ce, weighted_kd = (1-kd_lambda)*ce, kd_lambda*kd
    return VanillaKDLoss(total=weighted_ce+weighted_kd, ce=ce, kd=kd, weighted_ce=weighted_ce, weighted_kd=weighted_kd)


def _with_auxiliary(base: VanillaKDLoss, auxiliary: torch.Tensor, weight: float) -> VanillaKDLoss:
    if weight < 0:
        raise ValueError("Auxiliary KD weight must be non-negative.")
    kd = base.kd + weight * auxiliary
    weighted_kd = (base.weighted_kd / (base.kd + 1e-12)) * kd
    return VanillaKDLoss(total=base.weighted_ce + weighted_kd, ce=base.ce, kd=kd,
                         weighted_ce=base.weighted_ce, weighted_kd=weighted_kd)


def attention_transfer_loss(student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    """Normalized attention-map transfer for shape-matched terminal stages."""
    s = student_feature.square().mean(dim=1).flatten(1)
    t = teacher_feature.square().mean(dim=1).flatten(1)
    return F.mse_loss(F.normalize(s, dim=1), F.normalize(t, dim=1))


def relational_distance_loss(student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    """Scale-normalized pairwise distance geometry transfer (RKD-style)."""
    s = student_feature.flatten(1); t = teacher_feature.flatten(1)
    sd = torch.cdist(s, s, p=2); td = torch.cdist(t, t, p=2)
    return F.smooth_l1_loss(sd / (sd.detach().mean() + 1e-12), td / (td.detach().mean() + 1e-12))


@torch.no_grad()
def ternarized_feature_target(feature: torch.Tensor, factor: float = 0.7) -> torch.Tensor:
    """Per-sample/channel ternary activation proxy used only as frozen QFD target."""
    scale = feature.abs().mean(dim=(2, 3), keepdim=True)
    threshold = factor * scale
    code = torch.where(feature > threshold, torch.ones_like(feature), torch.where(feature < -threshold, -torch.ones_like(feature), torch.zeros_like(feature)))
    active = feature.abs() * (code != 0)
    alpha = active.sum(dim=(2, 3), keepdim=True) / (code != 0).sum(dim=(2, 3), keepdim=True).clamp_min(1)
    return code * alpha


def quantized_feature_loss(student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    """QFD screen: match a frozen ternarized teacher terminal feature target."""
    target = ternarized_feature_target(teacher_feature)
    return F.mse_loss(F.normalize(student_feature.flatten(1), dim=1), F.normalize(target.flatten(1), dim=1))


def quantized_relational_loss(student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    """QTRD screen: preserve geometry of a frozen ternarized teacher feature."""
    return relational_distance_loss(student_feature, ternarized_feature_target(teacher_feature))


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
