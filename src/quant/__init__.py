"""Ternary quantization layers used by the Task 3 QAT pipeline."""

from .ternary import (
    QuantConfig,
    TernaryConv2d,
    TernaryLinear,
    TernaryQuantizer,
    convert_to_ternary,
    count_ternary_layers,
    ternary_parameter_groups,
)

__all__ = [
    "QuantConfig", "TernaryConv2d", "TernaryLinear", "TernaryQuantizer",
    "convert_to_ternary", "count_ternary_layers", "ternary_parameter_groups",
]
