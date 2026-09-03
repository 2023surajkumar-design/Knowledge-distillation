"""
src/quant/ste.py  —  Straight-Through Estimators for ternary quantization.

The ternary assignment Q(W) ∈ {-1, 0, +1} is a hard threshold, so it has zero
gradient almost everywhere. We use a Straight-Through Estimator (STE) to pass a
surrogate gradient from the quantized output back to the latent FP32 weight.

Two variants are provided (default = clipped):

  * identity STE  — dL/dW = dL/dQ everywhere (Bengio et al., 2013).
  * clipped  STE  — dL/dW = dL/dQ · 1[|W| ≤ clip]  (default clip = 1.0).
                    Yin et al. (ICLR 2019) show a bounded/clipped estimator has
                    a coarse gradient positively correlated with the true one
                    and gives convergent training, unlike the raw identity STE
                    which can be unstable near minima. This is our justified
                    default for the Task 3 baseline; richer variants (ReSTE etc.)
                    are deferred to the Task 7 STE study.

The estimator is applied to the W -> Q(W) map only. The scale α (learned or
derived) multiplies Q(W) *outside* this Function, so autograd differentiates the
scale exactly while the STE only governs the discrete threshold.
"""

from __future__ import annotations

import torch


class TernarizeSTE(torch.autograd.Function):
    """Forward: hard ternary sign Q(W) ∈ {-1,0,+1} given a (broadcastable) delta.
    Backward: straight-through (optionally clipped) gradient to W."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, delta: torch.Tensor,
                clip: float, use_clip: bool) -> torch.Tensor:
        q = torch.zeros_like(w)
        q = torch.where(w > delta, torch.ones_like(w), q)
        q = torch.where(w < -delta, -torch.ones_like(w), q)
        ctx.save_for_backward(w)
        ctx.clip = clip
        ctx.use_clip = use_clip
        return q

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (w,) = ctx.saved_tensors
        grad_w = grad_out
        if ctx.use_clip:
            grad_w = grad_w * (w.abs() <= ctx.clip).to(grad_out.dtype)
        # grads for (w, delta, clip, use_clip)
        return grad_w, None, None, None


def ternarize_ste(w: torch.Tensor, delta: torch.Tensor,
                  ste: str = "clipped", clip: float = 1.0) -> torch.Tensor:
    """Return Q(W) ∈ {-1,0,+1} with the chosen straight-through backward."""
    use_clip = (ste == "clipped")
    if ste not in ("clipped", "identity"):
        raise ValueError(f"Unknown STE variant '{ste}' (expected 'clipped' or 'identity').")
    return TernarizeSTE.apply(w, delta, clip, use_clip)
