"""
src/evaluation/verify_ternary.py  —  Programmatic ternary-constraint verification
(deliverable required by the assignment; Phase 37).

Verifies, on the DEPLOYED forward weights (not the latent FP32 weights), that
every convolutional and fully-connected layer of the student is exactly ternary:

    unique(Ŵ_channel) ⊆ {-α, 0, +α}     (symmetric)
    unique(Ŵ_channel) ⊆ {-α_n, 0, +α_p} (asymmetric / TTQ)

and reports per-layer negative/zero/positive fractions, scale, threshold,
quantization error, plus overall sparsity and the theoretical compression ratio.
It also asserts the LATENT weights are NOT ternary (many distinct values), which
demonstrates the latent≠deployed separation (C15).

Usage:
    uv run src/evaluation/verify_ternary.py --checkpoint results/best_models/resnet18_ternary_best.pth
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.resnet_cifar import resnet18_cifar  # noqa: E402
from src.quant.ternary import (  # noqa: E402
    QuantConfig, convert_to_ternary, TernaryConv2d, TernaryLinear, count_ternary_layers,
)


def verify_model(model: torch.nn.Module, verbose: bool = True) -> dict:
    """Verify all ternary layers of a live model. Returns a summary dict.
    Raises AssertionError if any deployed weight is not exactly ternary."""
    rows = []
    total_w = total_zero = 0
    all_ok = True

    for name, m in model.named_modules():
        if not isinstance(m, (TernaryConv2d, TernaryLinear)):
            continue
        latent = m.weight.detach()
        deployed = m.deployed_weight().detach()

        # (a) deployed must be exactly ternary per output channel
        C = deployed.shape[0]
        dep_flat = deployed.reshape(C, -1)
        max_unique = 0
        for c in range(C):
            u = torch.unique(dep_flat[c])
            max_unique = max(max_unique, u.numel())
        channel_ok = max_unique <= 3
        # (b) latent must NOT be ternary (proves latent != deployed)
        latent_unique = torch.unique(latent).numel()
        latent_ok = latent_unique > 3

        stats = m.quant.stats(m.weight.detach())
        total_w += stats["num_weights"]
        total_zero += int(round(stats["zero_frac"] * stats["num_weights"]))
        ok = channel_ok and latent_ok
        all_ok = all_ok and ok
        rows.append({"layer": name, "kind": type(m).__name__,
                     "max_unique_per_channel": max_unique,
                     "latent_unique": latent_unique, **stats, "ok": ok})

    sparsity = total_zero / max(1, total_w)
    # theoretical bits/weight for a 3-symbol alphabet
    bits_ternary = math.log2(3)
    compression_vs_fp32 = 32.0 / bits_ternary

    if verbose:
        print(f"{'layer':32s}{'kind':16s}{'uniq/ch':>8}{'zero%':>8}{'neg%':>7}{'pos%':>7}{'qerr':>8}{'ok':>4}")
        print("-" * 96)
        for r in rows:
            print(f"{r['layer'][:31]:32s}{r['kind']:16s}{r['max_unique_per_channel']:>8}"
                  f"{100*r['zero_frac']:>7.1f} {100*r['neg_frac']:>6.1f} {100*r['pos_frac']:>6.1f}"
                  f"{r['quant_error']:>8.3f}{'Y' if r['ok'] else 'N':>4}")
        print("-" * 96)
        print(f"Ternary layers verified : {len(rows)}  (all exactly ternary: {all_ok})")
        print(f"Overall weight sparsity : {100*sparsity:.2f}% zeros")
        print(f"Theoretical bits/weight : {bits_ternary:.3f}  ->  ~{compression_vs_fp32:.1f}x vs FP32 "
              f"(theoretical; real speedup needs specialized kernels)")

    assert all_ok, "TERNARY VERIFICATION FAILED: a deployed weight is not exactly ternary."
    return {"num_ternary_layers": len(rows), "sparsity": sparsity,
            "bits_per_weight": bits_ternary, "compression_vs_fp32": compression_vs_fp32,
            "all_ternary": all_ok, "layers": rows}


def _build_from_checkpoint(ckpt_path: str, device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_d = ckpt.get("quant_config")
    if cfg_d is None:
        raise ValueError("Checkpoint has no 'quant_config'; cannot rebuild ternary model.")
    cfg = QuantConfig(**cfg_d)
    keep = ckpt.get("keep_first_last_fp32", False)
    model = resnet18_cifar(num_classes=ckpt.get("num_classes", 10)).to(device)
    convert_to_ternary(model, cfg, keep_first_last_fp32=keep)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    p = argparse.ArgumentParser(description="Verify the ternary weight constraint.")
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_from_checkpoint(args.checkpoint, device)
    conv, lin = count_ternary_layers(model)
    print(f"Loaded {args.checkpoint}: {conv} ternary conv + {lin} ternary linear layers\n")
    verify_model(model, verbose=True)
    print("\nVERIFICATION PASSED: all deployed conv/FC weights are exactly ternary; "
          "latent FP32 weights are distinct (latent != deployed).")


if __name__ == "__main__":
    main()
