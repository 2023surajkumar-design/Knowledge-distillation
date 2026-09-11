from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "final"
PLOT = ROOT / "plots" / "final"
OUT.mkdir(parents=True, exist_ok=True)
PLOT.mkdir(parents=True, exist_ok=True)


def read(path: Path):
    return json.loads(path.read_text())


def summary(path: str):
    return read(ROOT / path)


def row(name, architecture, precision, kd, status, path, ternary=None, sparsity=None, quant_error=None, compression=None):
    data = summary(path)
    per_seed = data.get("per_seed", [])
    best_epochs = [x.get("best_epoch") for x in per_seed if x.get("best_epoch") is not None]
    return {
        "model": name,
        "architecture": architecture,
        "deployment": precision,
        "kd_method": kd,
        "status": status,
        "seeds": [x.get("seed") for x in per_seed],
        "validation_mean": data.get("validation_mean"),
        "validation_std": data.get("validation_std"),
        "best_validation": data.get("validation_best", data.get("best_validation_accuracy")),
        "best_epoch": max(best_epochs) if best_epochs else None,
        "strict_ternary": data.get("all_conv_and_fc_ternary", ternary),
        "sparsity": sparsity,
        "quant_error": quant_error,
        "theoretical_compression": compression,
        "test_evaluation": data.get("test_evaluation"),
        "source": path,
    }


rows = [
    row("FP32 ResNet-18", "ResNet-18", "FP32", "none", "confirmed", "results/task2_final_summary.json"),
    row("Ternary ResNet-18", "ResNet-18", "strict ternary QAT", "none", "confirmed", "results/task3_b2_default_summary.json", True, 0.4714, 0.5589, 20.2),
    row("Vanilla ternary KD", "ResNet-18", "strict ternary QAT", "vanilla KD", "confirmed control", "results/task4/task4_b3_final_t2_lam09_remaining_rerun1_summary.json", True, 0.4768, 0.5616, 20.2),
    row("DKD", "ResNet-18", "strict ternary QAT", "DKD", "rejected", "results/task4/task5_dkd_final_t2_lam09_a1_b8_r1_summary.json", True, 0.4767, None, 20.2),
    row("DIST", "ResNet-18", "strict ternary QAT", "DIST", "screening", "results/task4/task6_dist_screen_t2_lam09_i1_a1_r1_summary.json", True, None, None, 20.2),
]
# Task 1 has a preserved teacher checkpoint/history but no standalone summary
# JSON; retain the documented validation reference and label its provenance.
rows.insert(0, {
    "model": "FP32 ResNet-34 teacher",
    "architecture": "ResNet-34",
    "deployment": "FP32",
    "kd_method": "none",
    "status": "confirmed reference",
    "seeds": [],
    "validation_mean": 0.9554,
    "validation_std": None,
    "best_validation": 0.9554,
    "best_epoch": None,
    "strict_ternary": False,
    "sparsity": None,
    "quant_error": None,
    "theoretical_compression": 1.0,
    "test_evaluation": "not_run",
    "source": "resnet34_cifar10_fp32_best.pth + documented Task 1 validation metadata",
})
# The saved continuation summary covers seeds 43/44; the authoritative control
# aggregate also includes the preserved seed-42 anchor.
rows[3]["validation_mean"] = 0.9512
rows[3]["validation_std"] = 0.00151
rows[3]["best_validation"] = 0.9526

comparison_path = OUT / "final_comparison.json"
comparison_path.write_text(json.dumps({"generated": datetime.now().astimezone().isoformat(), "test_evaluation": "LOCKED_NOT_RUN", "rows": rows}, indent=2) + "\n")

# Accuracy comparison.
labels = [r["model"] for r in rows]
means = [100 * float(r["validation_mean"]) for r in rows]
stds = [100 * float(r["validation_std"] or 0) for r in rows]
colors = ["#264653", "#245b8a", "#2a9d8f", "#457b9d", "#c44536", "#e09f3e"]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(labels, means, yerr=stds, capsize=4, color=colors)
ax.set_ylabel("Validation accuracy (%)")
ax.set_ylim(90, 96)
ax.set_title("ATDL validation comparison (official test locked)")
ax.tick_params(axis="x", rotation=18)
fig.tight_layout()
fig.savefig(PLOT / "validation_comparison.png", dpi=180)
plt.close(fig)

# Curves from real histories where available.
curve_sources = {
    "Task 3 ternary": ROOT / "experiments/histories/task3_b2_default/resnet18_ternary_seed44.json",
    "Task 4 vanilla KD": ROOT / "experiments/task4/histories/task4_b3_final_t2_lam09_remaining_rerun1/seed44.json",
    "DKD": ROOT / "experiments/task4/histories/task5_dkd_final_t2_lam09_a1_b8_r1/seed42.json",
    "DIST": ROOT / "experiments/task4/histories/task6_dist_screen_t2_lam09_i1_a1_r1/seed42.json",
}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for label, path in curve_sources.items():
    data = read(path)
    hist = data.get("history", {})
    val = hist.get("val_acc", hist.get("validation_accuracy", []))
    loss = hist.get("val_loss", hist.get("validation_loss", []))
    if val:
        axes[0].plot(range(1, len(val) + 1), [100 * x for x in val], label=label)
    if loss:
        axes[1].plot(range(1, len(loss) + 1), loss, label=label)
axes[0].set_title("Validation accuracy trajectories")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[1].set_title("Validation loss trajectories")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(PLOT / "training_validation_curves.png", dpi=180)
plt.close(fig)

# Compression calculation for ResNet-18 parameters.
fp32_params = 11_173_962
ternary_bits = fp32_params * math.log2(3) + 64 * 32
compression = (fp32_params * 32) / ternary_bits
compression_data = {
    "parameter_count": fp32_params,
    "fp32_bits": fp32_params * 32,
    "ideal_ternary_bits_plus_64_scales": ternary_bits,
    "theoretical_ratio": compression,
    "sparsity_reference": 0.4714,
    "caveat": "The ratio excludes packing/runtime overhead and does not imply hardware speedup.",
}
(OUT / "compression_analysis.json").write_text(json.dumps(compression_data, indent=2) + "\n")

# Compact report source and PDF.
report = f"""# ATDL Final Validation Report

Generated: {datetime.now().astimezone().isoformat()}

## Scope and firewall

This report uses the fixed CIFAR-10 45k/5k validation split. The official test set was never used; all source summaries report `test_evaluation: not_run`.

## Final comparison

| Model | Validation mean | Std | Best | Status |
|---|---:|---:|---:|---|
""" + "\n".join(f"| {r['model']} | {100*r['validation_mean']:.2f}% | {100*(r['validation_std'] or 0):.2f}% | {100*(r['best_validation'] or 0):.2f}% | {r['status']} |" for r in rows) + f"""

## Conclusions

- The completed vanilla Task 4 control is the strongest verified ternary KD candidate at 95.12% validation mean.
- DKD is rejected as the final candidate: its three-seed mean is 94.88%, below Task 4.
- DIST remains a one-seed screening result at 94.94%, below Task 4, and is not promoted.
- The ternary no-KD Task 3 control remains a strong reference at approximately 95.21% mean.
- No new method was trained in the finalization pass because no supported remaining method had evidence sufficient to justify GPU expenditure under the time constraint.

## Quantization and compression

The mainline student uses strict ternary Conv/FC deployment with latent FP32 training parameters and QAT/STE. The Task 3 reference reports approximately 47.14% sparsity and 0.5589 mean quantization error. The ideal ternary storage estimate is approximately {compression:.1f}x before scale metadata, packing, and runtime overhead. This is not a measured hardware speedup.

## Diagnostics

Existing per-seed diagnostics include CE/KD contributions, teacher/student confidence, entropy, agreement, gradient norms/cosine, learning-rate curves, checkpoint reload checks, sparsity, and quantization error where logged. Missing metrics are not fabricated. Consolidated plots are in `plots/final/`.

## Reproducibility

Best verified final control checkpoint: `experiments/task4/checkpoints/task4_b3_final_t2_lam09_remaining_rerun1/resnet18_ternary_kd_seed44.pth`.

AutoResearch/AutoML remains paused, not abandoned. Future work should resume with one separately reviewed one-seed experiment targeting quantization-compatible knowledge transfer, preserving the test firewall and append-only artifacts.
"""
(OUT / "final_validation_report.md").write_text(report)

with PdfPages(OUT / "final_validation_report.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.04, 0.96, "ATDL Final Validation Report", fontsize=20, weight="bold", va="top")
    ax.text(0.04, 0.915, "Validation-only; official CIFAR-10 test set locked", fontsize=11, va="top")
    y = 0.85
    for r in rows:
        line = f"{r['model']}: {100*r['validation_mean']:.2f}% +/- {100*(r['validation_std'] or 0):.2f}% ({r['status']})"
        ax.text(0.06, y, line, fontsize=11, va="top")
        y -= 0.045
    ax.text(0.04, y - 0.02, "Conclusion", fontsize=14, weight="bold", va="top")
    ax.text(0.06, y - 0.07, "Vanilla Task 4 remains the strongest verified ternary KD control.\nDKD is rejected; DIST remains screening-only.\nNo official test metrics were computed.", fontsize=11, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    for image in (PLOT / "validation_comparison.png", PLOT / "training_validation_curves.png"):
        fig = plt.figure(figsize=(11, 7))
        fig.figimage(plt.imread(image), xo=0, yo=0)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(json.dumps({"comparison": str(comparison_path.relative_to(ROOT)), "report_md": str((OUT / 'final_validation_report.md').relative_to(ROOT)), "report_pdf": str((OUT / 'final_validation_report.pdf').relative_to(ROOT)), "plots": sorted(str(p.relative_to(ROOT)) for p in PLOT.glob('*.png'))}, indent=2))
