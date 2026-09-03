#!/usr/bin/env python3
"""
scripts/train_student_ternary.py  —  TASK 3: ternary ResNet18 QAT baseline (B2).

Trains a ResNet18 whose ALL conv + FC WEIGHTS are ternary {-α,0,+α}, with the
ternary constraint ACTIVE during training (genuine QAT, not post-hoc PTQ).
NO knowledge distillation is used in Task 3 — this is the pure-quantization
reference that isolates the cost of ternarization (FP32 ResNet18 -> ternary).

This same script is the host for Tasks 4/5/6 (KD): the `--kd-mode` flag exists
for interface compatibility but any value other than 'none' raises here, because
KD is Task 4.

Design (see src/quant/ternary.py):
  * latent FP32 weight per layer; forward uses the quantized weight (C15).
  * default quantizer: per-channel symmetric TWN (derived α), clipped STE.
  * warm-start latent weights from the Task 2 FP32 checkpoint by default (TTQ
    practice — improves QAT stability); disable with --no-warm-start.
  * weight decay applied to latent conv/linear weights only.

Selection is on the validation split; the test set is never touched here.
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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data import get_train_val_loaders, CIFAR10_MEAN, CIFAR10_STD  # noqa: E402
from src.models.resnet_cifar import resnet18_cifar  # noqa: E402
from src.quant.ternary import (  # noqa: E402
    QuantConfig, convert_to_ternary, ternary_parameter_groups, count_ternary_layers,
)
from src.evaluation.verify_ternary import verify_model  # noqa: E402
from src.training.utils import (  # noqa: E402
    set_seed, get_device, make_warmup_cosine, train_one_epoch, evaluate,
    save_checkpoint, save_history, save_training_curves,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x=None, *a, **k):
        return x


def parse_args():
    p = argparse.ArgumentParser(description="Task 3 — ternary ResNet18 QAT baseline (no KD)")
    # data / optim
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--scale-lr-with-batch", action="store_true", default=True)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="on latent weights only")
    p.add_argument("--nesterov", action="store_true", default=True)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--strict-determinism", action="store_true", default=False)
    # quantizer config (Task 3 ablation knobs)
    p.add_argument("--threshold", choices=["twn", "ttq"], default="twn")
    p.add_argument("--scale", choices=["derived", "learned"], default="derived")
    p.add_argument("--symmetric", dest="symmetric", action="store_true", default=True)
    p.add_argument("--asymmetric", dest="symmetric", action="store_false")
    p.add_argument("--scope", choices=["per_channel", "per_tensor"], default="per_channel")
    p.add_argument("--t", type=float, default=0.05, help="TTQ threshold factor")
    p.add_argument("--ste", choices=["clipped", "identity"], default="clipped")
    p.add_argument("--clip", type=float, default=1.0)
    # DIAGNOSTIC ONLY — must be False for the compliant baseline (C14)
    p.add_argument("--keep-first-last-fp32", action="store_true", default=False,
                   help="[diagnostic] leave first conv + FC in FP32 for a sensitivity probe.")
    # warm start
    p.add_argument("--warm-start", default="results/best_models/resnet18_fp32_best.pth")
    p.add_argument("--no-warm-start", dest="warm_start", action="store_const", const="")
    # io
    p.add_argument("--ckpt-dir", default="experiments/checkpoints")
    p.add_argument("--hist-dir", default="experiments/histories")
    p.add_argument("--plot-dir", default="plots")
    p.add_argument("--best-dir", default="results/best_models")
    p.add_argument("--summary", default="results/resnet18_ternary_noKD_summary.json")
    # scope guards
    p.add_argument("--kd-mode", default="none",
                   help="[Task 4+ placeholder] must be 'none' in Task 3.")
    p.add_argument("--smoke", action="store_true", default=False)
    return p.parse_args()


def build_ternary_student(cfg: QuantConfig, warm_start: str, keep_fl_fp32: bool, device):
    model = resnet18_cifar(num_classes=10)
    warmed = False
    if warm_start:
        if os.path.exists(warm_start):
            ck = torch.load(warm_start, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model_state_dict"])  # FP32 weights into latent
            warmed = True
        else:
            print(f"[warn] warm-start checkpoint not found ({warm_start}); training from scratch.")
    convert_to_ternary(model, cfg, keep_first_last_fp32=keep_fl_fp32)
    return model.to(device), warmed


def train_one_seed(seed, args, cfg, device):
    set_seed(seed, strict=args.strict_determinism)
    eff_lr = args.lr * args.batch_size / 128.0 if args.scale_lr_with_batch else args.lr
    epochs = 1 if args.smoke else args.epochs

    train_loader, val_loader, tr_idx, va_idx = get_train_val_loaders(
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, loader_seed=seed)

    model, warmed = build_ternary_student(cfg, args.warm_start, args.keep_first_last_fp32, device)
    nconv, nlin = count_ternary_layers(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        ternary_parameter_groups(model, args.weight_decay),
        lr=eff_lr, momentum=args.momentum, nesterov=args.nesterov)
    scheduler = make_warmup_cosine(optimizer, args.warmup_epochs, epochs)

    config = {
        "task": "T3_ternary_resnet18_baseline_noKD", "arch": "ResNet18-CIFAR-ternary",
        "seed": seed, "epochs": epochs, "batch_size": args.batch_size,
        "effective_lr": eff_lr, "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs, "label_smoothing": args.label_smoothing,
        "warm_started_from_fp32": warmed, "num_classes": 10,
        "normalization": {"mean": list(CIFAR10_MEAN), "std": list(CIFAR10_STD)},
        "ternary_conv_layers": nconv, "ternary_linear_layers": nlin,
    }
    quant_config = vars(cfg).copy()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "gap": []}
    best_val, best_epoch = 0.0, -1
    ckpt_path = os.path.join(args.ckpt_dir, f"resnet18_ternary_noKD_seed{seed}.pth")

    print(f"\n=== Seed {seed} | ternary ResNet18 (no KD) | {epochs} ep | eff_lr={eff_lr:.4f} "
          f"| quant={cfg.scope}/{cfg.threshold}/{'sym' if cfg.symmetric else 'asym'}/{cfg.scale}/STE={cfg.ste} "
          f"| ternary layers={nconv}conv+{nlin}fc | warm_start={warmed} ===")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            progress=lambda it: tqdm(it, desc=f"S{seed} Ep{epoch}/{epochs}", leave=False))
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        gap = tr_acc - va_acc
        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc), ("val_loss", va_loss),
                     ("val_acc", va_acc), ("lr", cur_lr), ("gap", gap)]:
            history[k].append(v)
        if not (np.isfinite(tr_loss) and np.isfinite(va_loss)):
            raise RuntimeError(f"NaN/inf loss at epoch {epoch} — QAT diverged (lower LR or check STE).")
        improved = va_acc > best_val
        if improved:
            best_val, best_epoch = va_acc, epoch
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val, config,
                            extra={"quant_config": quant_config,
                                   "keep_first_last_fp32": args.keep_first_last_fp32})
        print(f"S{seed} Ep{epoch:3d}/{epochs} | train {tr_loss:.4f}/{100*tr_acc:5.2f}% | "
              f"val {va_loss:.4f}/{100*va_acc:5.2f}% | gap {100*gap:5.2f}% | lr {cur_lr:.5f}"
              + ("  <-- best" if improved else ""))

    elapsed = time.time() - t0
    save_history(os.path.join(args.hist_dir, f"resnet18_ternary_noKD_seed{seed}.json"),
                 history, {"best_val_acc": best_val, "best_epoch": best_epoch,
                           "config": config, "quant_config": quant_config, "elapsed_sec": elapsed})
    save_training_curves(history, os.path.join(args.plot_dir, f"resnet18_ternary_noKD_seed{seed}"))

    # Verify the saved best model is exactly ternary (no test set involved).
    best = resnet18_cifar(10).to(device)
    convert_to_ternary(best, cfg, keep_first_last_fp32=args.keep_first_last_fp32)
    best.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    vinfo = verify_model(best, verbose=False)
    print(f"Seed {seed} done in {elapsed/60:.1f} min | best val {100*best_val:.2f}% @ ep {best_epoch} "
          f"| sparsity {100*vinfo['sparsity']:.1f}% | all_ternary={vinfo['all_ternary']}")

    return {"seed": seed, "best_val_acc": best_val, "best_epoch": best_epoch, "ckpt": ckpt_path,
            "final_train_acc": history["train_acc"][-1], "sparsity": vinfo["sparsity"],
            "all_ternary": vinfo["all_ternary"], "elapsed_sec": elapsed}


def main():
    args = parse_args()
    if args.kd_mode != "none":
        raise NotImplementedError(
            "KD is Task 4, not Task 3. `--kd-mode` exists only for interface compatibility. "
            "Run Task 3 with --kd-mode none (default)."
        )
    cfg = QuantConfig(threshold=args.threshold, scale=args.scale, symmetric=args.symmetric,
                      scope=args.scope, t=args.t, ste=args.ste, clip=args.clip).validate()

    device = get_device()
    print("==== Task 3: ternary ResNet18 QAT baseline (no KD) ====")
    print(f"Python {platform.python_version()} | torch {torch.__version__} | device {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Quant config: {vars(cfg)}")

    results = [train_one_seed(s, args, cfg, device) for s in args.seeds]
    val_accs = np.array([r["best_val_acc"] for r in results])
    overall_best = max(results, key=lambda r: r["best_val_acc"])

    os.makedirs(args.best_dir, exist_ok=True)
    best_out = os.path.join(args.best_dir, "resnet18_ternary_noKD_best.pth")
    torch.save(torch.load(overall_best["ckpt"], map_location="cpu", weights_only=False), best_out)

    summary = {
        "task": "T3_ternary_resnet18_baseline_noKD", "seeds": args.seeds,
        "quant_config": vars(cfg),
        "val_acc_mean": float(val_accs.mean()), "val_acc_std": float(val_accs.std()),
        "val_acc_best": float(val_accs.max()), "best_seed": overall_best["seed"],
        "best_checkpoint": best_out, "sparsity_mean": float(np.mean([r["sparsity"] for r in results])),
        "all_ternary": all(r["all_ternary"] for r in results), "per_seed": results,
        "note": "Validation results only. Compare against Task 2 (FP32 ResNet18) to isolate the "
                "ternarization cost. Final test via scripts/evaluate.py --final-evaluation.",
    }
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== Task 3 summary (validation) ====")
    print(f"Val acc: {100*val_accs.mean():.2f}% ± {100*val_accs.std():.2f} "
          f"(best {100*val_accs.max():.2f}%, seed {overall_best['seed']})")
    print(f"Mean sparsity: {100*summary['sparsity_mean']:.1f}%  | all layers ternary: {summary['all_ternary']}")
    print(f"Best checkpoint -> {best_out}\nSummary -> {args.summary}")
    print("Verify: uv run src/evaluation/verify_ternary.py --checkpoint " + best_out)


if __name__ == "__main__":
    main()
