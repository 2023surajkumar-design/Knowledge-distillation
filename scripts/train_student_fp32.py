#!/usr/bin/env python3
"""Task 2 B1: FP32 ResNet-18 trained from scratch, with no KD or quantization."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CIFAR10_MEAN, CIFAR10_STD, get_train_val_loaders, split_manifest
from src.models.resnet_cifar import count_parameters, resnet18_cifar
from src.training.utils import (
    evaluate,
    get_device,
    make_warmup_cosine,
    preview_warmup_cosine,
    runtime_metadata,
    save_checkpoint,
    save_json,
    save_training_curves,
    set_seed,
    train_one_epoch,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


def load_baseline_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    required = {
        "task": "T2_fp32_resnet18_baseline", "arch": "ResNet18-CIFAR", "num_classes": 10,
        "dataset": "CIFAR-10", "optimizer": "SGD", "scheduler": "warmup+cosine",
        "label_smoothing": 0.1, "kd": False, "quantization": "none", "allow_tf32": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Baseline config must set {key}={expected!r}; got {config.get(key)!r}.")
    return config


def parse_args() -> tuple[argparse.Namespace, dict, Path]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="configs/student/resnet18_fp32.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    config_path = Path(pre_args.config)
    config = load_baseline_config(config_path)

    parser = argparse.ArgumentParser(description="Task 2 B1: FP32 ResNet18 baseline")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--data-root", default=config["data_root"])
    parser.add_argument("--epochs", type=int, default=config["epochs"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--lr", type=float, default=config["lr"])
    parser.add_argument("--scale-lr-with-batch", action=argparse.BooleanOptionalAction,
                        default=config["scale_lr_with_batch"])
    parser.add_argument("--momentum", type=float, default=config["momentum"])
    parser.add_argument("--weight-decay", type=float, default=config["weight_decay"])
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=config["nesterov"])
    parser.add_argument("--warmup-epochs", type=int, default=config["warmup_epochs"])
    parser.add_argument("--label-smoothing", type=float, default=config["label_smoothing"])
    parser.add_argument("--num-workers", type=int, default=config["num_workers"])
    parser.add_argument("--seeds", type=int, nargs="+", default=config["seeds"])
    parser.add_argument("--strict-determinism", action=argparse.BooleanOptionalAction,
                        default=config["strict_determinism"])
    parser.add_argument("--run-name", default=None, help="immutable artifact namespace")
    parser.add_argument("--smoke", action="store_true", help="mark a short real-CIFAR preflight run")
    parser.add_argument("--kd", action="store_true", help="Task 4 placeholder; prohibited in Task 2")
    args = parser.parse_args()
    if args.kd:
        raise NotImplementedError("KD is Task 4, not Task 2; run this baseline without --kd.")
    if args.epochs < 1 or args.warmup_epochs < 0 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive; warmup epochs cannot be negative.")
    if args.smoke and args.epochs >= config["epochs"]:
        raise ValueError("--smoke requires an explicit shorter --epochs value (e.g. --epochs 5).")
    if args.run_name is None:
        args.run_name = f"task2_preflight_{args.epochs}e" if args.smoke else "task2_b1_fp32_resnet18"
    return args, config, config_path


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "checkpoint_dir": Path("experiments/checkpoints") / args.run_name,
        "history_dir": Path("experiments/histories") / args.run_name,
        "plot_dir": Path("plots") / args.run_name,
        "best_checkpoint": Path("results/best_models") / f"{args.run_name}_best.pth",
        "summary": (Path("results/preflight") / f"{args.run_name}.json") if args.smoke
        else Path("results/task2_final_summary.json"),
    }


def assert_fresh_outputs(paths: dict[str, Path], seeds: list[int]) -> None:
    targets = [paths["best_checkpoint"], paths["summary"]]
    targets += [paths["checkpoint_dir"] / f"resnet18_fp32_seed{seed}.pth" for seed in seeds]
    targets += [paths["history_dir"] / f"resnet18_fp32_seed{seed}.json" for seed in seeds]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite immutable experiment artifacts. Choose a new --run-name: " + ", ".join(existing)
        )


def train_one_seed(
    seed: int, args: argparse.Namespace, baseline_config: dict, config_path: Path,
    device: torch.device, paths: dict[str, Path],
) -> dict:
    set_seed(seed, strict=args.strict_determinism)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    effective_lr = args.lr * args.batch_size / 128.0 if args.scale_lr_with_batch else args.lr
    train_loader, val_loader, train_indices, _ = get_train_val_loaders(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, loader_seed=seed,
    )
    model = resnet18_cifar(num_classes=10).to(device)
    if count_parameters(model) != 11_173_962:
        raise AssertionError("Unexpected ResNet18 parameter count.")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=effective_lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=args.nesterov,
    )
    scheduler = make_warmup_cosine(optimizer, args.warmup_epochs, args.epochs)
    manifest = split_manifest(train_loader.dataset.dataset.targets)
    config = {
        "task": baseline_config["task"], "arch": baseline_config["arch"], "dataset": "CIFAR-10",
        "run_name": args.run_name, "seed": seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "base_lr": args.lr, "effective_lr": effective_lr, "scale_lr_with_batch": args.scale_lr_with_batch,
        "momentum": args.momentum, "weight_decay": args.weight_decay, "nesterov": args.nesterov,
        "warmup_epochs": args.warmup_epochs, "scheduler": "warmup+cosine",
        "label_smoothing": args.label_smoothing, "strict_determinism": args.strict_determinism,
        "allow_tf32": False, "num_workers": args.num_workers, "num_classes": 10,
        "normalization": {"mean": list(CIFAR10_MEAN), "std": list(CIFAR10_STD)},
        "split_manifest": manifest, "runtime": runtime_metadata(device), "config_source": str(config_path),
        "run_kind": "preflight" if args.smoke else "B1_control",
    }
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "gap": []}
    best_val_accuracy, best_epoch = -float("inf"), -1
    checkpoint_path = paths["checkpoint_dir"] / f"resnet18_fp32_seed{seed}.pth"
    print(
        f"\n=== {args.run_name} | seed {seed} | {args.epochs} epochs | LR {effective_lr:.5f} | "
        f"params {count_parameters(model):,} ==="
    )
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            progress=lambda iterator: tqdm(iterator, desc=f"Seed {seed} epoch {epoch}/{args.epochs}", leave=False),
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss); history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_accuracy)
        history["lr"].append(current_lr); history["gap"].append(train_accuracy - val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy, best_epoch = val_accuracy, epoch
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_accuracy, config)
        print(
            f"seed {seed} epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f}/{100*train_accuracy:5.2f}% | "
            f"val {val_loss:.4f}/{100*val_accuracy:5.2f}% | aug-train−val {100*(train_accuracy-val_accuracy):5.2f}pp | "
            f"lr {current_lr:.6f}" + ("  <-- best" if val_accuracy == best_val_accuracy else "")
        )
    elapsed_seconds = time.perf_counter() - start_time
    peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    # Selection remains the in-run validation score. This independent reload check
    # catches serialization/architecture mistakes without using test data. In the
    # default fast CUDA mode, allow one 5k-validation example at a decision boundary.
    reloaded = resnet18_cifar(num_classes=10).to(device)
    reloaded_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(reloaded_checkpoint["model_state_dict"], strict=True)
    reloaded_validation_loss, reloaded_validation_accuracy = evaluate(reloaded, val_loader, criterion, device)
    reload_accuracy_delta = abs(reloaded_validation_accuracy - best_val_accuracy)
    if reload_accuracy_delta > 1 / 5_000 + 1e-12:
        raise RuntimeError(
            "Reloaded validation accuracy differs by more than one validation example; "
            "refusing to accept this checkpoint."
        )
    history_path = paths["history_dir"] / f"resnet18_fp32_seed{seed}.json"
    plot_prefix = paths["plot_dir"] / f"resnet18_fp32_seed{seed}"
    save_json(history_path, {
        "history": history, "best_validation_accuracy": best_val_accuracy, "best_epoch": best_epoch,
        "config": config, "elapsed_seconds": elapsed_seconds,
        "seconds_per_epoch": elapsed_seconds / args.epochs, "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "checkpoint_reload_validation_accuracy": reloaded_validation_accuracy,
        "checkpoint_reload_validation_loss": reloaded_validation_loss,
        "checkpoint_reload_accuracy_delta": reload_accuracy_delta,
    })
    plot_paths = save_training_curves(history, plot_prefix)
    return {
        "seed": seed, "best_validation_accuracy": best_val_accuracy, "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path), "history": str(history_path), "plots": plot_paths,
        "elapsed_seconds": elapsed_seconds, "seconds_per_epoch": elapsed_seconds / args.epochs,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes, "final_augmented_train_accuracy": history["train_acc"][-1],
        "final_validation_accuracy": history["val_acc"][-1], "n_train": len(train_indices),
        "checkpoint_reload_validation_accuracy": reloaded_validation_accuracy,
        "checkpoint_reload_accuracy_delta": reload_accuracy_delta,
    }


def main() -> None:
    args, baseline_config, config_path = parse_args()
    paths = paths_for(args)
    assert_fresh_outputs(paths, args.seeds)
    device = get_device(require_cuda=True)
    lr_preview = preview_warmup_cosine(
        args.lr * args.batch_size / 128.0 if args.scale_lr_with_batch else args.lr,
        args.warmup_epochs, args.epochs,
    )
    print(f"Task 2 {args.run_name}: CUDA {torch.cuda.get_device_name(device)}")
    print(f"LR epochs 1..7: {[round(value, 8) for value in lr_preview[:7]]}; epoch {args.epochs}: {lr_preview[-1]:.10f}")
    results = [train_one_seed(seed, args, baseline_config, config_path, device, paths) for seed in args.seeds]
    validation_accuracies = np.asarray([result["best_validation_accuracy"] for result in results])
    overall_best = max(results, key=lambda result: result["best_validation_accuracy"])
    paths["best_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(overall_best["checkpoint"], paths["best_checkpoint"])
    summary = {
        "task": "T2", "condition": "B1_FP32_ResNet18", "run_name": args.run_name,
        "architecture": "ResNet18-CIFAR", "parameters": 11_173_962, "dataset": "CIFAR-10",
        "train_samples": 45_000, "validation_samples": 5_000, "test_samples": 10_000,
        "validation_per_class": 500, "training_seeds": args.seeds, "epochs": args.epochs,
        "batch_size": args.batch_size, "optimizer": "SGD", "momentum": args.momentum,
        "nesterov": args.nesterov, "weight_decay": args.weight_decay, "base_lr": args.lr,
        "effective_lr": args.lr * args.batch_size / 128.0 if args.scale_lr_with_batch else args.lr,
        "warmup_epochs": args.warmup_epochs, "scheduler": "warmup+cosine",
        "label_smoothing": args.label_smoothing,
        "augmentation": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip"],
        "test_evaluation": "not_run", "test_mean": None, "test_std": None,
        "validation_mean": float(validation_accuracies.mean()),
        "validation_std": float(validation_accuracies.std(ddof=1)) if len(validation_accuracies) > 1 else 0.0,
        "validation_best": float(validation_accuracies.max()), "best_seed": overall_best["seed"],
        "best_checkpoint": str(paths["best_checkpoint"]), "per_seed": results,
        "note": "Validation-only summary. The augmented-training-minus-validation metric is a diagnostic, not a clean generalization gap.",
    }
    save_json(paths["summary"], summary)
    print(f"Validation: {100*summary['validation_mean']:.2f}% ± {100*summary['validation_std']:.2f}%")
    print(f"Best validation checkpoint: {paths['best_checkpoint']}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    main()
