"""Straight-through estimators for the hard ternary assignment."""

from __future__ import annotations

import torch


class TernarizeSTE(torch.autograd.Function):
    """Hard ternarization in the forward pass and a selected coarse gradient."""

    @staticmethod
    def forward(
        ctx, weight: torch.Tensor, delta: torch.Tensor, clip: float, use_clip: bool
    ) -> torch.Tensor:
        quantized = torch.zeros_like(weight)
        quantized = torch.where(weight > delta, torch.ones_like(weight), quantized)
        quantized = torch.where(weight < -delta, -torch.ones_like(weight), quantized)
        ctx.save_for_backward(weight)
        ctx.clip = clip
        ctx.use_clip = use_clip
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight,) = ctx.saved_tensors
        gradient = grad_output
        if ctx.use_clip:
            gradient = gradient * (weight.abs() <= ctx.clip).to(grad_output.dtype)
        # The threshold is a non-learned statistic in Task 3; no surrogate gradient.
        return gradient, None, None, None


def ternarize_ste(
    weight: torch.Tensor, delta: torch.Tensor, ste: str = "clipped", clip: float = 1.0
) -> torch.Tensor:
    """Return {-1, 0, +1} while passing identity or clipped STE gradients to ``weight``."""
    if ste not in {"clipped", "identity"}:
        raise ValueError(f"Unknown STE variant {ste!r}.")
    if clip <= 0:
        raise ValueError("STE clip must be positive.")
    return TernarizeSTE.apply(weight, delta, float(clip), ste == "clipped")
