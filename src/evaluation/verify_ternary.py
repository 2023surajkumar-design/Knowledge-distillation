"""Verify the deployed (not latent) strict ternary constraint for Task 3."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.resnet_cifar import resnet18_cifar
from src.quant.ternary import QuantConfig, TernaryConv2d, TernaryLinear, convert_to_ternary


def _expected_layer_kinds(model: nn.Module) -> tuple[int, int]:
    return (
        sum(isinstance(module, nn.Conv2d) for module in model.modules()),
        sum(isinstance(module, nn.Linear) for module in model.modules()),
    )


@torch.no_grad()
def verify_model(model: nn.Module, require_all_weight_layers: bool = True, verbose: bool = True) -> dict:
    """Assert that each ternary layer's actual deployed values match its quantizer.

    ``require_all_weight_layers`` is true for the official B2 run.  Diagnostic
    first/last-FP32 probes can set it false, but are then explicitly labelled
    non-compliant in their result artifacts.
    """
    rows: list[dict] = []
    total_weights = total_zeros = 0
    ternary_conv = ternary_linear = 0
    all_ok = True
    for name, layer in model.named_modules():
        if not isinstance(layer, (TernaryConv2d, TernaryLinear)):
            continue
        ternary_conv += isinstance(layer, TernaryConv2d)
        ternary_linear += isinstance(layer, TernaryLinear)
        latent = layer.weight.detach()
        code = layer.quant.ternary_code(latent)
        expected = layer.quant.deployed_from_code(code, latent)
        deployed = layer.deployed_weight().detach()
        # Direct equality to the quantizer's allowed per-group values is stronger
        # than merely testing ``unique_count <= 3``.
        deployed_matches = bool(torch.allclose(deployed, expected, rtol=0.0, atol=1e-7))
        code_ok = bool(torch.all((code == -1) | (code == 0) | (code == 1)))
        flat = deployed.reshape(deployed.shape[0], -1)
        max_unique = max(torch.unique(channel).numel() for channel in flat)
        latent_unique = torch.unique(latent).numel()
        stats = layer.quant.stats(latent)
        gradient_norm = None if layer.weight.grad is None else float(layer.weight.grad.detach().norm().item())
        ok = deployed_matches and code_ok and max_unique <= 3 and latent_unique > 3
        all_ok = all_ok and ok
        total_weights += stats["num_weights"]
        total_zeros += int(round(stats["zero_frac"] * stats["num_weights"]))
        rows.append({
            "layer": name, "kind": type(layer).__name__, "output_channels": deployed.shape[0],
            "max_unique_per_channel": max_unique, "latent_unique": latent_unique,
            "deployed_matches_quantizer": deployed_matches, "gradient_norm": gradient_norm,
            **stats, "ok": ok,
        })

    expected_conv, expected_linear = _expected_layer_kinds(model)
    all_weight_layers_ternary = ternary_conv == expected_conv and ternary_linear == expected_linear
    compliant = all_ok and (all_weight_layers_ternary or not require_all_weight_layers)
    if require_all_weight_layers and not all_weight_layers_ternary:
        all_ok = False
    if not compliant:
        raise AssertionError("TERNARY VERIFICATION FAILED: deployed values or layer coverage are non-compliant.")
    bits_per_weight = math.log2(3)
    output = {
        "num_ternary_layers": len(rows), "ternary_conv_layers": ternary_conv,
        "ternary_linear_layers": ternary_linear, "expected_conv_layers": expected_conv,
        "expected_linear_layers": expected_linear, "all_weight_layers_ternary": all_weight_layers_ternary,
        "all_ternary": all_ok, "sparsity": total_zeros / max(total_weights, 1),
        "bits_per_weight": bits_per_weight, "compression_vs_fp32": 32 / bits_per_weight,
        "layers": rows,
    }
    if verbose:
        print(f"Ternary layer coverage: {ternary_conv} conv + {ternary_linear} FC; "
              f"all conv/FC ternary={all_weight_layers_ternary}")
        print(f"Deployed sparsity: {100 * output['sparsity']:.2f}% | theoretical compression: "
              f"{output['compression_vs_fp32']:.2f}x before scale overhead")
        for row in rows:
            print(f"{row['layer']:30s} zero={100*row['zero_frac']:5.1f}% "
                  f"qerr={row['quant_error']:.3f} unique/ch={row['max_unique_per_channel']} ok={row['ok']}")
    return output


def build_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("quant_config")
    if not config:
        raise ValueError("Checkpoint lacks quant_config; exact ternary reconstruction is impossible.")
    if checkpoint.get("arch") != "ResNet18-CIFAR-ternary":
        raise ValueError(f"Unexpected architecture {checkpoint.get('arch')!r}.")
    model = resnet18_cifar(num_classes=int(checkpoint.get("num_classes", 10))).to(device)
    convert_to_ternary(
        model, QuantConfig(**config), keep_first_last_fp32=bool(checkpoint.get("keep_first_last_fp32", False))
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify actual deployed ternary ResNet18 weights.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", help="Optional JSON diagnostics output; never accesses test data.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_from_checkpoint(args.checkpoint, device)
    info = verify_model(model, require_all_weight_layers=not checkpoint.get("keep_first_last_fp32", False))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print("VERIFICATION PASSED: latent FP32 parameters and deployed ternary weights are distinct.")


if __name__ == "__main__":
    main()
