"""Strict ternary weight layers and conversion utilities for Task 3 QAT.

Each ternary layer owns a trainable FP32 latent ``weight``.  Its forward pass
*always* uses the deployed weight ``W_hat = alpha * Q(W_latent)``; the latent
weight is never supplied to convolution or linear operators.  BatchNorm and
bias parameters intentionally remain FP32.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ste import ternarize_ste


@dataclass(frozen=True)
class QuantConfig:
    threshold: str = "twn"
    scale: str = "derived"
    symmetric: bool = True
    scope: str = "per_channel"
    t: float = 0.05
    twn_factor: float = 0.7
    ste: str = "clipped"
    clip: float = 1.0

    def validate(self) -> "QuantConfig":
        if self.threshold not in {"twn", "ttq"}:
            raise ValueError("threshold must be 'twn' or 'ttq'.")
        if self.scale not in {"derived", "learned"}:
            raise ValueError("scale must be 'derived' or 'learned'.")
        if self.scope not in {"per_channel", "per_tensor"}:
            raise ValueError("scope must be 'per_channel' or 'per_tensor'.")
        if self.ste not in {"clipped", "identity"} or self.clip <= 0:
            raise ValueError("STE must be clipped/identity with a positive clip.")
        if not 0 < self.t < 1 or not 0 < self.twn_factor < 2:
            raise ValueError("Threshold factors must be finite positive fractions.")
        if not self.symmetric and self.scale != "learned":
            raise ValueError("Asymmetric ternary requires learned positive and negative scales.")
        return self

    def serializable(self) -> dict:
        return asdict(self)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus for positive-scale initialization."""
    return value + torch.log(-torch.expm1(-value))


class TernaryQuantizer(nn.Module):
    """Quantizes a latent weight tensor into its deployed three-level representation."""

    _MIN_SCALE = 1e-6

    def __init__(self, weight_shape: torch.Size, cfg: QuantConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.out_channels = int(weight_shape[0])
        self.ndim = len(weight_shape)
        self.n_groups = self.out_channels if cfg.scope == "per_channel" else 1
        self._view = (self.n_groups,) + (1,) * (self.ndim - 1)
        if cfg.scale == "learned":
            initial = torch.ones(self.n_groups)
            if cfg.symmetric:
                self.raw_alpha = nn.Parameter(_inverse_softplus(initial))
            else:
                self.raw_alpha_p = nn.Parameter(_inverse_softplus(initial))
                self.raw_alpha_n = nn.Parameter(_inverse_softplus(initial))

    def _grouped(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.reshape(self.out_channels, -1) if self.cfg.scope == "per_channel" else weight.reshape(1, -1)

    def _statistics_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Threshold and derived-scale statistics are deliberately FP32 under AMP.
        return weight.detach().float()

    def _delta(self, weight: torch.Tensor) -> torch.Tensor:
        grouped_abs = self._grouped(self._statistics_weight(weight)).abs()
        if self.cfg.threshold == "twn":
            delta = self.cfg.twn_factor * grouped_abs.mean(dim=1)
        else:
            delta = self.cfg.t * grouped_abs.max(dim=1).values
        return delta.view(self._view).to(device=weight.device, dtype=weight.dtype)

    def _derived_alpha(self, weight: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        grouped_abs = self._grouped(self._statistics_weight(weight)).abs()
        grouped_delta = delta.detach().float().reshape(self.n_groups, 1)
        active = grouped_abs > grouped_delta
        alpha = (grouped_abs * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
        # A completely zero group must remain exactly zero rather than manufacture scale.
        return alpha.view(self._view).to(device=weight.device, dtype=weight.dtype)

    def _learned_scales(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.symmetric:
            alpha = (F.softplus(self.raw_alpha) + self._MIN_SCALE).view(self._view)
            alpha = alpha.to(dtype=dtype)
            return alpha, alpha
        positive = (F.softplus(self.raw_alpha_p) + self._MIN_SCALE).view(self._view).to(dtype=dtype)
        negative = (F.softplus(self.raw_alpha_n) + self._MIN_SCALE).view(self._view).to(dtype=dtype)
        return positive, negative

    @torch.no_grad()
    def init_learned_from_weight(self, weight: torch.Tensor) -> None:
        if self.cfg.scale != "learned":
            return
        alpha = self._derived_alpha(weight, self._delta(weight)).reshape(self.n_groups).float()
        raw = _inverse_softplus(alpha.clamp_min(self._MIN_SCALE))
        if self.cfg.symmetric:
            self.raw_alpha.copy_(raw)
        else:
            self.raw_alpha_p.copy_(raw)
            self.raw_alpha_n.copy_(raw)

    def ternary_code(self, weight: torch.Tensor) -> torch.Tensor:
        """Hard code used by verification; intentionally has no STE/autograd semantics."""
        delta = self._delta(weight)
        return torch.where(weight > delta, torch.ones_like(weight), torch.where(weight < -delta, -torch.ones_like(weight), torch.zeros_like(weight)))

    def deployed_from_code(self, code: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.cfg.scale == "derived":
            alpha = self._derived_alpha(weight, self._delta(weight)).detach()
            return code * alpha
        alpha_p, alpha_n = self._learned_scales(code.dtype)
        return torch.where(code > 0, code * alpha_p, torch.where(code < 0, code * alpha_n, code))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        code = ternarize_ste(weight, self._delta(weight), ste=self.cfg.ste, clip=self.cfg.clip)
        return self.deployed_from_code(code, weight)

    @torch.no_grad()
    def stats(self, weight: torch.Tensor) -> dict:
        code = self.ternary_code(weight)
        deployed = self.deployed_from_code(code, weight)
        n_weights = weight.numel()
        difference = weight.detach().float() - deployed.detach().float()
        stats = {
            "num_weights": n_weights,
            "neg_frac": float((code < 0).float().mean().item()),
            "zero_frac": float((code == 0).float().mean().item()),
            "pos_frac": float((code > 0).float().mean().item()),
            "delta_mean": float(self._delta(weight).float().mean().item()),
            "delta_min": float(self._delta(weight).float().min().item()),
            "delta_max": float(self._delta(weight).float().max().item()),
            "quant_error": float(difference.norm().item() / (weight.detach().float().norm().item() + 1e-12)),
            "mae": float(difference.abs().mean().item()),
            "mse": float(difference.square().mean().item()),
            "latent_mean": float(weight.detach().float().mean().item()),
            "latent_std": float(weight.detach().float().std(unbiased=False).item()),
            "latent_min": float(weight.detach().float().min().item()),
            "latent_max": float(weight.detach().float().max().item()),
        }
        if self.cfg.scale == "derived":
            alpha = self._derived_alpha(weight, self._delta(weight)).float()
            stats.update(alpha_mean=float(alpha.mean().item()), alpha_min=float(alpha.min().item()), alpha_max=float(alpha.max().item()))
        else:
            alpha_p, alpha_n = self._learned_scales(torch.float32)
            stats.update(alpha_p_mean=float(alpha_p.mean().item()), alpha_n_mean=float(alpha_n.mean().item()))
            if self.cfg.symmetric:
                stats.update(alpha_mean=float(alpha_p.mean().item()), alpha_min=float(alpha_p.min().item()), alpha_max=float(alpha_p.max().item()))
        return stats


class TernaryConv2d(nn.Conv2d):
    def __init__(self, *args, quant_cfg: Optional[QuantConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = TernaryQuantizer(self.weight.shape, quant_cfg or QuantConfig())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(inputs, self.quant(self.weight), self.bias)

    @torch.no_grad()
    def deployed_weight(self) -> torch.Tensor:
        return self.quant.deployed_from_code(self.quant.ternary_code(self.weight), self.weight)


class TernaryLinear(nn.Linear):
    def __init__(self, *args, quant_cfg: Optional[QuantConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = TernaryQuantizer(self.weight.shape, quant_cfg or QuantConfig())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.quant(self.weight), self.bias)

    @torch.no_grad()
    def deployed_weight(self) -> torch.Tensor:
        return self.quant.deployed_from_code(self.quant.ternary_code(self.weight), self.weight)


def _replace_conv(module: nn.Conv2d, cfg: QuantConfig) -> TernaryConv2d:
    replacement = TernaryConv2d(
        module.in_channels, module.out_channels, module.kernel_size, stride=module.stride,
        padding=module.padding, dilation=module.dilation, groups=module.groups,
        bias=module.bias is not None, padding_mode=module.padding_mode, quant_cfg=cfg,
    )
    replacement = replacement.to(device=module.weight.device, dtype=module.weight.dtype)
    replacement.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        replacement.bias.data.copy_(module.bias.data)
    replacement.quant.init_learned_from_weight(replacement.weight)
    return replacement


def _replace_linear(module: nn.Linear, cfg: QuantConfig) -> TernaryLinear:
    replacement = TernaryLinear(module.in_features, module.out_features, bias=module.bias is not None, quant_cfg=cfg)
    replacement = replacement.to(device=module.weight.device, dtype=module.weight.dtype)
    replacement.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        replacement.bias.data.copy_(module.bias.data)
    replacement.quant.init_learned_from_weight(replacement.weight)
    return replacement


def convert_to_ternary(
    model: nn.Module,
    cfg: QuantConfig,
    first_conv_name: str = "conv1",
    last_fc_name: str = "fc",
    keep_first_last_fp32: bool = False,
) -> nn.Module:
    """Replace every Conv2d/Linear, or only the explicitly named diagnostic exclusions."""
    cfg.validate()
    replacements: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)) or isinstance(module, (TernaryConv2d, TernaryLinear)):
            continue
        # Exact names matter: ``layer1.0.conv1`` is not the model stem.
        if keep_first_last_fp32 and name in {first_conv_name, last_fc_name}:
            continue
        replacements.append((name, module))
    for name, module in replacements:
        parent = model
        *parents, attribute = name.split(".")
        for item in parents:
            parent = getattr(parent, item)
        setattr(parent, attribute, _replace_conv(module, cfg) if isinstance(module, nn.Conv2d) else _replace_linear(module, cfg))
    return model


def count_ternary_layers(model: nn.Module) -> tuple[int, int]:
    return (
        sum(isinstance(module, TernaryConv2d) for module in model.modules()),
        sum(isinstance(module, TernaryLinear) for module in model.modules()),
    )


def ternary_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay only latent Conv/Linear weights; leave BN, bias, and learned scales undecayed."""
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    decay_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.requires_grad
    }
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if id(parameter) in decay_ids else no_decay).append(parameter)
    if len({id(parameter) for parameter in decay + no_decay}) != len(decay) + len(no_decay):
        raise AssertionError("Optimizer parameter grouping contains duplicates.")
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def parameter_group_audit(model: nn.Module, groups: Iterable[dict]) -> dict:
    """Small machine-readable invariant report used by the smoke test and notebook."""
    decay_ids = {id(parameter) for group in groups if group["weight_decay"] > 0 for parameter in group["params"]}
    expected_ids = {id(module.weight) for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))}
    return {
        "decayed_parameter_tensors": len(decay_ids),
        "expected_latent_weight_tensors": len(expected_ids),
        "only_conv_linear_latents_decayed": decay_ids == expected_ids,
        "bn_bias_and_scale_tensors_decayed": any(
            id(parameter) in decay_ids and parameter.ndim < 2 for parameter in model.parameters()
        ),
    }
