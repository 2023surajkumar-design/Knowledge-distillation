#!/usr/bin/env python3
"""
scripts/train_student_fp32.py  —  TASK 2: FP32 ResNet18 baseline (B1).

Trains a CIFAR-adapted ResNet18 in full precision, from scratch, with NO KD and
NO quantization, using the SAME recipe and the SAME frozen 45k/5k split as the
Task 1 teacher, so the FP32 teacher-vs-student capacity gap is a controlled
comparison.

Scope guard (KD.md "execute one task at a time"):
  * This script implements TASK 2 ONLY.
  * It never touches the test set (selection is on the validation split).
    Final test numbers come from scripts/evaluate.py --final-evaluation.
  * A `--kd` flag exists for interface compatibility with later tasks but is a
    placeholder: enabling it raises, because KD is Task 4.

Deliverables produced:
  * per-seed best checkpoint  -> experiments/checkpoints/resnet18_fp32_seed{S}.pth
  * overall best (max val acc) -> results/best_models/resnet18_fp32_best.pth
  * per-seed history JSON      -> experiments/histories/resnet18_fp32_seed{S}.json
  * training curves (PNG)      -> plots/resnet18_fp32_seed{S}_*.png
  * aggregate summary          -> results/resnet18_fp32_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# --- make `src` importable whether run from repo root or scripts/ ------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data import get_train_val_loaders, CIFAR10_MEAN, CIFAR10_STD  # noqa: E402
from src.models.resnet_cifar import resnet18_cifar, count_parameters  # noqa: E402
from src.training.utils import (  # noqa: E402
    set_seed, get_device, make_warmup_cosine, train_one_epoch, evaluate,
    save_checkpoint, save_history, save_training_curves,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x=None, *a, **k):
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2 — FP32 ResNet18 baseline")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="base LR for batch size 128")
    p.add_argument("--scale-lr-with-batch", action="store_true", default=True)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--nesterov", action="store_true", default=True)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--strict-determinism", action="store_true", default=False)
    p.add_argument("--ckpt-dir", default="experiments/checkpoints")
    p.add_argument("--hist-dir", default="experiments/histories")
    p.add_argument("--plot-dir", default="plots")
    p.add_argument("--best-dir", default="results/best_models")
    p.add_argument("--summary", default="results/resnet18_fp32_summary.json")
    # interface-compat placeholder — do NOT enable in Task 2
    p.add_argument("--kd", action="store_true", default=False,
                   help="[Task 4 placeholder] KD is not implemented in Task 2.")
    # fast pipeline check
    p.add_argument("--smoke", action="store_true", default=False,
                   help="1-epoch run for pipeline validation (not a real result).")
    return p.parse_args()


def train_one_seed(seed: int, args: argparse.Namespace, device: torch.device) -> dict:
    set_seed(seed, strict=args.strict_determinism)

    eff_lr = args.lr * args.batch_size / 128.0 if args.scale_lr_with_batch else args.lr
    epochs = 1 if args.smoke else args.epochs

    train_loader, val_loader, train_idx, val_idx = get_train_val_loaders(
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, loader_seed=seed)

    model = resnet18_cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=eff_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = make_warmup_cosine(optimizer, args.warmup_epochs, epochs)

    config = {
        "task": "T2_fp32_resnet18_baseline", "arch": "ResNet18-CIFAR",
        "seed": seed, "epochs": epochs, "batch_size": args.batch_size,
        "base_lr": args.lr, "effective_lr": eff_lr, "momentum": args.momentum,
        "weight_decay": args.weight_decay, "nesterov": args.nesterov,
        "warmup_epochs": args.warmup_epochs, "label_smoothing": args.label_smoothing,
        "normalization": {"mean": list(CIFAR10_MEAN), "std": list(CIFAR10_STD)},
        "num_classes": 10, "split": "stratified 45k/5k (seed 42)",
        "n_train": len(train_idx), "n_val": len(val_idx),
    }

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "lr": [], "gap": []}
    best_val, best_epoch = 0.0, -1
    ckpt_path = os.path.join(args.ckpt_dir, f"resnet18_fp32_seed{seed}.pth")

    print(f"\n=== Seed {seed} | ResNet18 FP32 | {epochs} epochs | eff_lr={eff_lr:.4f} "
          f"| params={count_parameters(model):,} ===")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            progress=lambda it: tqdm(it, desc=f"Seed {seed} Ep {epoch}/{epochs}", leave=False))
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        gap = tr_acc - va_acc
        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("val_loss", va_loss), ("val_acc", va_acc),
                     ("lr", cur_lr), ("gap", gap)]:
            history[k].append(v)

        improved = va_acc > best_val
        if improved:
            best_val, best_epoch = va_acc, epoch
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val, config)

        print(f"Seed {seed} Ep {epoch:3d}/{epochs} | "
              f"train {tr_loss:.4f}/{100*tr_acc:5.2f}% | "
              f"val {va_loss:.4f}/{100*va_acc:5.2f}% | gap {100*gap:5.2f}% | "
              f"lr {cur_lr:.5f}" + ("  <-- best" if improved else ""))

    elapsed = time.time() - t0
    save_history(os.path.join(args.hist_dir, f"resnet18_fp32_seed{seed}.json"),
                 history, {"best_val_acc": best_val, "best_epoch": best_epoch,
                           "config": config, "elapsed_sec": elapsed})
    save_training_curves(history, os.path.join(args.plot_dir, f"resnet18_fp32_seed{seed}"))
    print(f"Seed {seed} done in {elapsed/60:.1f} min | best val {100*best_val:.2f}% @ ep {best_epoch}")

    return {"seed": seed, "best_val_acc": best_val, "best_epoch": best_epoch,
            "ckpt": ckpt_path, "final_train_acc": history["train_acc"][-1],
            "elapsed_sec": elapsed}


def main() -> None:
    args = parse_args()
    if args.kd:
        raise NotImplementedError(
            "KD is Task 4, not Task 2. This flag exists only for interface "
            "compatibility with later student-training scripts. Run Task 2 without --kd."
        )

    device = get_device()
    print("==== Task 2: FP32 ResNet18 baseline ====")
    print(f"Python {platform.python_version()} | torch {torch.__version__} | device {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = [train_one_seed(s, args, device) for s in args.seeds]

    val_accs = np.array([r["best_val_acc"] for r in results])
    overall_best = max(results, key=lambda r: r["best_val_acc"])

    # Copy the overall-best-on-val checkpoint to the canonical name.
    os.makedirs(args.best_dir, exist_ok=True)
    best_out = os.path.join(args.best_dir, "resnet18_fp32_best.pth")
    torch.save(torch.load(overall_best["ckpt"], map_location="cpu", weights_only=False), best_out)

    summary = {
        "task": "T2_fp32_resnet18_baseline",
        "seeds": args.seeds,
        "val_acc_mean": float(val_accs.mean()),
        "val_acc_std": float(val_accs.std()),
        "val_acc_best": float(val_accs.max()),
        "best_seed": overall_best["seed"],
        "best_checkpoint": best_out,
        "per_seed": results,
        "note": "Validation-set results only. Run scripts/evaluate.py "
                "--final-evaluation for the official test number (final reporting).",
    }
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== Task 2 summary (validation) ====")
    print(f"Val acc: {100*val_accs.mean():.2f}% ± {100*val_accs.std():.2f} "
          f"(best {100*val_accs.max():.2f}%, seed {overall_best['seed']})")
    print(f"Best checkpoint -> {best_out}")
    print(f"Summary -> {args.summary}")
    print("NEXT: scripts/evaluate.py --checkpoint <best> --final-evaluation  (final report only)")


if __name__ == "__main__":
    main()
