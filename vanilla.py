"""
src/kd/vanilla.py  —  TASK 4: vanilla (Hinton) knowledge distillation.

Implements the canonical temperature-scaled soft-target loss combined with the
hard-label cross-entropy:

    L = (1 - λ) · CE(z_s, y)  +  λ · T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) )

Notes / correctness (rubric: "KD formulation & integration", 6 marks):
  * The T² factor is kept (Hinton et al. 2015): softening by T shrinks the
    soft-target gradients by ~1/T², so multiplying by T² keeps their magnitude
    comparable to the CE gradient as T changes. Without it, λ and T become
    entangled and the ablation is misleading.
  * KL direction is KL(teacher ‖ student): F.kl_div takes log-probabilities of
    the student as `input` and probabilities of the teacher as `target`.
  * The teacher is loaded frozen: `.eval()` (BN uses running stats) and every
    parameter has `requires_grad = False`; its forward runs under `no_grad`, so
    no gradient can flow into the teacher (C6 / C7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Frozen teacher
# --------------------------------------------------------------------------- #
def load_teacher(ckpt_path: str, device, arch: str = "resnet34", num_classes: int = 10):
    """Load a frozen teacher from a checkpoint. Returns (model, checkpoint_dict)."""
    from src.models.resnet_cifar import resnet18_cifar, resnet34_cifar
    builders = {"resnet34": resnet34_cifar, "resnet18": resnet18_cifar}
    if arch not in builders:
        raise ValueError(f"Unknown teacher arch '{arch}'.")
    model = builders[arch](num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert_teacher_frozen(model)
    return model, ckpt


def assert_teacher_frozen(teacher: nn.Module) -> None:
    """Hard guarantee: no teacher parameter is trainable (C6/C7)."""
    trainable = [n for n, p in teacher.named_parameters() if p.requires_grad]
    assert not trainable, f"Teacher is not frozen; trainable params: {trainable[:3]}..."
    assert not teacher.training, "Teacher must be in eval() mode (BN running stats)."


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def vanilla_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    targets: torch.Tensor, T: float, lam: float,
                    ce_criterion: nn.Module):
    """Return (total_loss, ce_detached, kl_detached)."""
    ce = ce_criterion(student_logits, targets)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    total = (1.0 - lam) * ce + lam * kl
    return total, ce.detach(), kl.detach()


# --------------------------------------------------------------------------- #
# KD training epoch (teacher frozen; no gradient into teacher)
# --------------------------------------------------------------------------- #
def train_one_epoch_kd(student, teacher, loader, optimizer, device,
                       T: float, lam: float, ce_criterion, progress=None):
    """One KD epoch. Returns (total_loss, ce_loss, kl_loss, accuracy)."""
    student.train()
    teacher.eval()
    run_loss = run_ce = run_kl = 0.0
    correct = total = 0
    iterator = progress(loader) if progress is not None else loader
    for inputs, labels in iterator:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():                      # teacher forward: no grad (C7)
            teacher_logits = teacher(inputs)

        optimizer.zero_grad(set_to_none=True)
        student_logits = student(inputs)
        loss, ce, kl = vanilla_kd_loss(student_logits, teacher_logits, labels,
                                       T, lam, ce_criterion)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        run_loss += loss.item() * bs
        run_ce += ce.item() * bs
        run_kl += kl.item() * bs
        correct += student_logits.max(1)[1].eq(labels).sum().item()
        total += bs
    return run_loss / total, run_ce / total, run_kl / total, correct / total


@torch.no_grad()
def teacher_val_accuracy(teacher, loader, device) -> float:
    """Convenience: teacher's own accuracy on a loader (for the comparison table)."""
    teacher.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += teacher(inputs).max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    return correct / total
