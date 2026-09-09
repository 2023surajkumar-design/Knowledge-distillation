"""Small, evidence-labelled diagnostics for validation-only experiment results."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch


def _mean_quantization_error(summary: dict[str, Any]) -> float | None:
    values = [row.get("quant_error_mean") for row in summary.get("per_seed", [])]
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def diagnose(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Compare two result summaries without claiming causality.

    The output deliberately labels outcomes as observations and hypotheses. It
    is a triage aid, not an automatic decision-maker or significance test.
    """
    candidate_accuracy = float(candidate["validation_mean"])
    control_accuracy = float(control["validation_mean"])
    accuracy_delta = candidate_accuracy - control_accuracy
    candidate_error, control_error = _mean_quantization_error(candidate), _mean_quantization_error(control)
    candidate_sparsity = candidate.get("sparsity_mean")
    control_sparsity = control.get("sparsity_mean")
    observations = [f"validation_accuracy_delta={100 * accuracy_delta:+.3f} percentage points"]
    hypotheses: list[str] = []
    next_checks: list[str] = []
    if candidate_error is not None and control_error is not None:
        error_delta = candidate_error - control_error
        observations.append(f"mean_quantization_error_delta={error_delta:+.4f}")
        if accuracy_delta < 0 and error_delta > 0:
            hypotheses.append("Higher quantization error co-occurs with the validation regression.")
            next_checks.append("Inspect per-layer error, scale, threshold, and dead-channel diagnostics.")
    if candidate_sparsity is not None and control_sparsity is not None:
        sparsity_delta = float(candidate_sparsity) - float(control_sparsity)
        observations.append(f"sparsity_delta={100 * sparsity_delta:+.2f} percentage points")
        if accuracy_delta < 0 and abs(sparsity_delta) > 0.03:
            hypotheses.append("A large sparsity shift co-occurs with the validation regression.")
            next_checks.append("Check zero-weight balance before changing KD and quantization together.")
    if accuracy_delta < -0.001:
        hypotheses.append("Candidate underperforms its matched control; excessive KD pressure or optimisation mismatch is possible.")
        next_checks.append("Compare CE/KD weighted losses and CE/KD gradient cosine on a matched diagnostic run.")
    elif accuracy_delta > 0.001:
        hypotheses.append("Candidate has a screen-level validation gain; reproduction is required before attributing a mechanism.")
        next_checks.append("Promote with fixed controls and independent seed(s), not a wider unbounded search.")
    else:
        hypotheses.append("Difference is within one-tenth of a percentage point; treat as inconclusive until seed replication.")
        next_checks.append("Do not combine this component with another unproven component.")
    return {
        "candidate": candidate.get("run_name"), "control": control.get("run_name"),
        "observations": observations, "hypotheses": hypotheses, "next_checks": next_checks,
        "causal_status": "hypotheses only; no causal conclusion is inferred from summary metrics",
        "test_evaluation": "not_run",
    }


def gradient_loss_diagnostics(
    ce_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    total_loss: torch.Tensor,
    parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    layerwise: bool = False,
) -> dict[str, Any]:
    """Measure CE/KD/total gradient relationships without changing gradients.

    This diagnostic uses ``autograd.grad`` and never calls ``backward`` or an
    optimiser. It is optional because retaining three graphs can add noticeable
    overhead. Call it before the training backward pass, on a sampled batch.
    Results describe the sampled objective geometry only and cannot establish
    causality for a validation change.
    """
    named = [(name, parameter) for name, parameter in parameters if parameter.requires_grad]
    if not named:
        raise ValueError("No trainable parameters supplied for gradient diagnostics.")
    names, tensors = zip(*named)
    ce_grads = torch.autograd.grad(ce_loss, tensors, retain_graph=True, allow_unused=True)
    kd_grads = torch.autograd.grad(kd_loss, tensors, retain_graph=True, allow_unused=True)
    total_grads = torch.autograd.grad(total_loss, tensors, retain_graph=True, allow_unused=True)

    ce_sq = kd_sq = total_sq = dot = 0.0
    rows: list[dict[str, Any]] = []
    for name, ce_grad, kd_grad, total_grad in zip(names, ce_grads, kd_grads, total_grads):
        ce_norm = 0.0 if ce_grad is None else float(ce_grad.detach().float().norm().item())
        kd_norm = 0.0 if kd_grad is None else float(kd_grad.detach().float().norm().item())
        total_norm = 0.0 if total_grad is None else float(total_grad.detach().float().norm().item())
        local_dot = 0.0 if ce_grad is None or kd_grad is None else float((ce_grad.detach().float() * kd_grad.detach().float()).sum().item())
        ce_sq += ce_norm**2; kd_sq += kd_norm**2; total_sq += total_norm**2; dot += local_dot
        if layerwise:
            rows.append({
                "parameter": name, "gradient_norm_ce": ce_norm,
                "gradient_norm_kd": kd_norm, "gradient_norm_total": total_norm,
                "ce_kd_gradient_cosine": local_dot / (ce_norm * kd_norm + 1e-12),
            })
    ce_norm, kd_norm, total_norm = np.sqrt(ce_sq), np.sqrt(kd_sq), np.sqrt(total_sq)
    cosine = dot / (ce_norm * kd_norm + 1e-12)
    result: dict[str, Any] = {
        "gradient_norm_ce": ce_norm, "gradient_norm_kd": kd_norm,
        "gradient_norm_total": total_norm,
        "kd_ce_gradient_ratio": kd_norm / (ce_norm + 1e-12),
        "ce_kd_gradient_cosine": cosine,
        "gradient_alignment": "positive" if cosine > 0.05 else "negative" if cosine < -0.05 else "near_orthogonal",
        "causal_status": "sampled diagnostic only; requires repeated measurement and matched controls",
    }
    if layerwise:
        result["layers"] = rows
    return result


@torch.no_grad()
def representation_diagnostics(teacher_features: torch.Tensor, student_features: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
    """Return normalised feature alignment and optional class-separation summaries.

    Features must have the same flattened width; use this for matched-stage or
    dimension-free representations, not to silently project incompatible maps.
    """
    teacher = teacher_features.flatten(1).float()
    student = student_features.flatten(1).float()
    if teacher.shape != student.shape:
        raise ValueError(f"Feature shapes must match, got {teacher.shape} and {student.shape}.")
    cosine = torch.nn.functional.cosine_similarity(teacher, student, dim=1)
    output: dict[str, Any] = {
        "feature_cosine_mean": float(cosine.mean().item()),
        "feature_cosine_std": float(cosine.std(unbiased=False).item()),
        "teacher_feature_norm": float(teacher.norm(dim=1).mean().item()),
        "student_feature_norm": float(student.norm(dim=1).mean().item()),
    }
    if labels is not None:
        labels = labels.detach()
        classes = labels.unique()
        if classes.numel() > 1:
            teacher_centroids = torch.stack([teacher[labels == label].mean(0) for label in classes])
            student_centroids = torch.stack([student[labels == label].mean(0) for label in classes])
            output["class_centroid_cosine"] = float(torch.nn.functional.cosine_similarity(teacher_centroids, student_centroids, dim=1).mean().item())
    output["causal_status"] = "representation similarity is explanatory evidence, not an accuracy surrogate"
    return output


def correlation_diagnostics(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    """Return Pearson and Spearman association for Task-6 prioritisation only."""
    x_array, y_array = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    if x_array.shape != y_array.shape or x_array.size < 3:
        raise ValueError("Correlation requires paired arrays with at least three observations.")
    pearson = float(np.corrcoef(x_array, y_array)[0, 1])
    x_rank = np.argsort(np.argsort(x_array)); y_rank = np.argsort(np.argsort(y_array))
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return {"pearson": pearson, "spearman": spearman, "n": int(x_array.size), "causal_status": "association only; do not infer causal layer or sample weighting"}
