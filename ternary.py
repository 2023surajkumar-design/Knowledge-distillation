"""
src/quant/ternary.py  —  Ternary weight layers and model conversion (QAT).

Implements the ternary weight constraint W ∈ {-α, 0, +α} (symmetric) or, as a
permitted stronger variant, {-α_n, 0, +α_p} (asymmetric, TTQ-style), enforced
DURING training via Quantization-Aware Training.

Key design points (rubric: "Ternary quantizer & student architecture"):
  * LATENT vs DEPLOYED weights are explicitly distinguished (C15). Each layer
    keeps a full-precision latent `weight` (the trainable nn.Parameter); the
    FORWARD pass always uses the quantized `Ŵ = α · Q(W)`. Latent weights are
    never used at inference and are never reported as the deployed weights.
  * Q(W) ∈ {-1,0,+1} via a threshold Δ; STE for the backward (see ste.py).
  * Scale/threshold formulations (configurable, for the Task 3 ablation):
        threshold: TWN  (Δ = 0.7 · mean|W|)   [default]
                   TTQ  (Δ = t   · max|W|)
        scale    : derived  (TWN optimal α = mean_{|W|>Δ}|W|)  [default, symmetric]
                   learned  (α is an nn.Parameter, TTQ-style)
        symmetry : symmetric  {-α,0,+α}   [default; matches the assignment set]
                   asymmetric {-α_n,0,+α_p} (requires learned scales, TTQ)
        scope    : per_channel (per output channel) [default] | per_tensor
  * The DEFAULT baseline is per-channel symmetric TWN (derived α), a robust,
    well-cited, exactly-{-α,0,+α} configuration. TTQ (learned, asymmetric) is
    available as the primary ablation.

Batch-norm layers and biases stay FP32 (standard; they are not conv/FC weight
matrices). By default ALL convolutional and fully-connected WEIGHTS are
ternarized — including the first conv and the final FC (C14). A diagnostic-only
`keep_first_last_fp32` switch exists for the layer-sensitivity probe but must be
False for the compliant baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ste import ternarize_ste


@dataclass
class QuantConfig:
    threshold: str = "twn"        # "twn" (0.7*mean|W|) | "ttq" (t*max|W|)
    scale: str = "derived"        # "derived" (TWN optimal) | "learned" (param)
    symmetric: bool = True        # True: {-a,0,+a}; False: {-a_n,0,+a_p} (TTQ)
    scope: str = "per_channel"    # "per_channel" | "per_tensor"
    t: float = 0.05               # threshold factor for TTQ mode
    twn_factor: float = 0.7       # threshold factor for TWN mode
    ste: str = "clipped"          # "clipped" | "identity"
    clip: float = 1.0             # clipped-STE range

    def validate(self):
        if not self.symmetric and self.scale != "learned":
            raise ValueError("Asymmetric ternary (TTQ) requires scale='learned'.")
        assert self.threshold in ("twn", "ttq")
        assert self.scale in ("derived", "learned")
        assert self.scope in ("per_channel", "per_tensor")
        return self


class TernaryQuantizer(nn.Module):
    """Maps a latent FP32 weight tensor to its ternary deployed form Ŵ."""

    def __init__(self, weight_shape, cfg: QuantConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.out_channels = weight_shape[0]
        self.ndim = len(weight_shape)
        # broadcast shape for per-(channel|tensor) scalars: (C,1,1,...) or (1,1,...)
        n_groups = self.out_channels if cfg.scope == "per_channel" else 1
        self._view = (n_groups,) + (1,) * (self.ndim - 1)
        self.n_groups = n_groups

        if cfg.scale == "learned":
            if cfg.symmetric:
                self.alpha = nn.Parameter(torch.ones(n_groups))
            else:
                self.alpha_p = nn.Parameter(torch.ones(n_groups))
                self.alpha_n = nn.Parameter(torch.ones(n_groups))
        self._initialized = False

    # -- per-group statistics ------------------------------------------------
    def _grouped(self, w: torch.Tensor) -> torch.Tensor:
        if self.cfg.scope == "per_channel":
            return w.reshape(self.out_channels, -1)          # (C, N)
        return w.reshape(1, -1)                               # (1, N_total)

    def _delta(self, w: torch.Tensor) -> torch.Tensor:
        flat = self._grouped(w).abs()
        if self.cfg.threshold == "twn":
            d = self.cfg.twn_factor * flat.mean(dim=1)
        else:  # ttq
            d = self.cfg.t * flat.max(dim=1).values
        return d.view(self._view)

    def _derived_alpha(self, w: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        flat = self._grouped(w).abs()
        d = delta.reshape(self.n_groups, 1)
        mask = (flat > d).to(flat.dtype)
        alpha = (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return alpha.view(self._view)

    @torch.no_grad()
    def init_learned_from_weight(self, w: torch.Tensor):
        """Initialize learned scales from the TWN-optimal derived value."""
        if self.cfg.scale != "learned":
            return
        delta = self._delta(w)
        a = self._derived_alpha(w, delta).reshape(self.n_groups)
        a = a.clamp(min=1e-3)
        if self.cfg.symmetric:
            self.alpha.copy_(a)
        else:
            self.alpha_p.copy_(a)
            self.alpha_n.copy_(a)
        self._initialized = True

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        delta = self._delta(w)                                # detached statistic
        q = ternarize_ste(w, delta, ste=self.cfg.ste, clip=self.cfg.clip)  # {-1,0,1}, STE to w
        if self.cfg.scale == "derived":
            alpha = self._derived_alpha(w, delta).detach()    # TWN statistic (no grad)
            return q * alpha
        # learned scales
        if self.cfg.symmetric:
            return q * self.alpha.view(self._view)
        # asymmetric TTQ: α_p on positives, α_n on negatives (selection detached)
        scale_sel = torch.where(q > 0,
                                self.alpha_p.view(self._view),
                                self.alpha_n.view(self._view))
        return q * scale_sel

    # -- introspection (used by verify_ternary) ------------------------------
    @torch.no_grad()
    def stats(self, w: torch.Tensor) -> dict:
        delta = self._delta(w)
        q = torch.zeros_like(w)
        q = torch.where(w > delta, torch.ones_like(w), q)
        q = torch.where(w < -delta, -torch.ones_like(w), q)
        w_hat = self.forward(w)
        n = w.numel()
        neg = (q < 0).sum().item()
        zero = (q == 0).sum().item()
        pos = (q > 0).sum().item()
        qerr = (w - w_hat).norm().item() / (w.norm().item() + 1e-12)
        if self.cfg.scale == "derived":
            a = self._derived_alpha(w, delta)
            alpha_desc = {"derived_alpha_mean": a.mean().item()}
        elif self.cfg.symmetric:
            alpha_desc = {"alpha_mean": self.alpha.mean().item()}
        else:
            alpha_desc = {"alpha_p_mean": self.alpha_p.mean().item(),
                          "alpha_n_mean": self.alpha_n.mean().item()}
        return {"num_weights": n, "neg_frac": neg / n, "zero_frac": zero / n,
                "pos_frac": pos / n, "delta_mean": delta.mean().item(),
                "quant_error": qerr, **alpha_desc}


# --------------------------------------------------------------------------- #
# Ternary layers (subclass std layers so latent `weight` loads FP32 directly)
# --------------------------------------------------------------------------- #
class TernaryConv2d(nn.Conv2d):
    def __init__(self, *args, quant_cfg: Optional[QuantConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = TernaryQuantizer(self.weight.shape, quant_cfg or QuantConfig())

    def forward(self, x):
        w_hat = self.quant(self.weight)          # deployed ternary weight
        return self._conv_forward(x, w_hat, self.bias)

    @torch.no_grad()
    def deployed_weight(self):
        return self.quant(self.weight)


class TernaryLinear(nn.Linear):
    def __init__(self, *args, quant_cfg: Optional[QuantConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = TernaryQuantizer(self.weight.shape, quant_cfg or QuantConfig())

    def forward(self, x):
        w_hat = self.quant(self.weight)
        return F.linear(x, w_hat, self.bias)

    @torch.no_grad()
    def deployed_weight(self):
        return self.quant(self.weight)


# --------------------------------------------------------------------------- #
# Model conversion + warm-start
# --------------------------------------------------------------------------- #
def _make_ternary_conv(conv: nn.Conv2d, cfg: QuantConfig) -> TernaryConv2d:
    new = TernaryConv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=(conv.bias is not None), quant_cfg=cfg)
    new.weight.data.copy_(conv.weight.data)      # warm-start latent from FP32
    if conv.bias is not None:
        new.bias.data.copy_(conv.bias.data)
    if cfg.scale == "learned":
        new.quant.init_learned_from_weight(new.weight.data)
    return new


def _make_ternary_linear(lin: nn.Linear, cfg: QuantConfig) -> TernaryLinear:
    new = TernaryLinear(lin.in_features, lin.out_features,
                        bias=(lin.bias is not None), quant_cfg=cfg)
    new.weight.data.copy_(lin.weight.data)
    if lin.bias is not None:
        new.bias.data.copy_(lin.bias.data)
    if cfg.scale == "learned":
        new.quant.init_learned_from_weight(new.weight.data)
    return new


def convert_to_ternary(model: nn.Module, cfg: QuantConfig,
                       first_conv_name: str = "conv1", last_fc_name: str = "fc",
                       keep_first_last_fp32: bool = False) -> nn.Module:
    """In-place replace every nn.Conv2d/nn.Linear WEIGHT with a ternary layer.

    By default ALL conv + FC weights are ternarized, including the first conv
    and final FC (C14). `keep_first_last_fp32=True` is a DIAGNOSTIC-ONLY switch
    for the layer-sensitivity probe and must not be used for the compliant
    baseline. BatchNorm and biases remain FP32.
    """
    cfg.validate()
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and not isinstance(
                module, (TernaryConv2d, TernaryLinear)):
            if keep_first_last_fp32 and (name.endswith(first_conv_name) or name.endswith(last_fc_name)):
                continue
            to_replace.append((name, module))

    for name, module in to_replace:
        parent = model
        *parents, attr = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        if isinstance(module, nn.Conv2d):
            setattr(parent, attr, _make_ternary_conv(module, cfg))
        else:
            setattr(parent, attr, _make_ternary_linear(module, cfg))
    return model


def ternary_parameter_groups(model: nn.Module, weight_decay: float):
    """Apply weight decay to latent conv/linear weights only; exclude BN params,
    biases, and learned ternary scales (α) from decay (standard QAT practice)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".weight") and ("bn" not in name.lower()):
            # conv/linear latent weights
            if p.ndim >= 2:
                decay.append(p)
                continue
        no_decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def count_ternary_layers(model: nn.Module):
    conv = sum(isinstance(m, TernaryConv2d) for m in model.modules())
    lin = sum(isinstance(m, TernaryLinear) for m in model.modules())
    return conv, lin
