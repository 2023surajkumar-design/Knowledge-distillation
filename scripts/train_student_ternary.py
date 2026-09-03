#!/usr/bin/env python3
"""Task 3 only: strict ternary ResNet18 QAT with validation-only selection.

The script deliberately has no teacher, KD loss, or test-loader import.  It is
the single production training path called by the Task 3 notebook and screening
tools.  Every run has an explicit name and refuses to overwrite an existing
artifact, preserving the immutable B2-default result.
"""

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
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CIFAR10_MEAN, CIFAR10_STD, get_train_val_loaders, split_manifest
from src.evaluation.verify_ternary import build_from_checkpoint, verify_model
from src.models.resnet_cifar import resnet18_cifar
from src.quant.ternary import (
    QuantConfig, convert_to_ternary, count_ternary_layers, parameter_group_audit,
    ternary_parameter_groups,
)
from src.training.utils import (
    evaluate, get_device, make_warmup_cosine, runtime_metadata, save_checkpoint,
    save_json, save_training_curves, set_seed, train_one_epoch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 3 ternary ResNet18 QAT baseline (no KD).")
    parser.add_argument("--kd-mode", choices=["none", "vanilla"], default="none",
                        help="Use 'vanilla' only through the explicit Task 4 dispatcher.")
    parser.add_argument("--config", default="configs/ternary/resnet18_ternary_noKD.yaml")
    parser.add_argument("--run-name")
    parser.add_argument("--condition")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--threshold", choices=["twn", "ttq"])
    parser.add_argument("--scale", choices=["derived", "learned"])
    parser.add_argument("--scope", choices=["per_channel", "per_tensor"])
    parser.add_argument("--t", type=float)
    parser.add_argument("--twn-factor", type=float)
    parser.add_argument("--ste", choices=["clipped", "identity"])
    parser.add_argument("--clip", type=float)
    parser.add_argument("--symmetric", dest="symmetric", action="store_true", default=None)
    parser.add_argument("--asymmetric", dest="symmetric", action="store_false")
    parser.add_argument("--warm-start")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--keep-first-last-fp32", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run exactly one validation-only epoch.")
    parser.add_argument("--strict-determinism", action="store_true")
    return parser


def parse_args() -> tuple[argparse.Namespace, dict, Path]:
    parsed = _parser().parse_args()
    if parsed.kd_mode != "none":
        raise ValueError("Task 3 permits only --kd-mode none; vanilla dispatch should have occurred before parsing.")
    config_path = Path(parsed.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text())
    quant = dict(config["quant"])
    for key in ("threshold", "scale", "scope", "t", "twn_factor", "ste", "clip", "symmetric"):
        value = getattr(parsed, key.replace("_", "-"), None) if "-" in key else getattr(parsed, key, None)
        if value is not None:
            quant[key] = value
    config["quant"] = quant
    mapping = {
        "run_name": "run_name", "condition": "condition", "epochs": "epochs", "batch_size": "batch_size", "num_workers": "num_workers",
        "data_root": "data_root", "lr": "lr", "weight_decay": "weight_decay", "momentum": "momentum",
        "warmup_epochs": "warmup_epochs", "label_smoothing": "label_smoothing", "seeds": "seeds",
    }
    for argument, key in mapping.items():
        value = getattr(parsed, argument)
        if value is not None:
            config[key] = value
    if parsed.warm_start is not None:
        config["warm_start"] = parsed.warm_start
    if parsed.no_warm_start:
        config["warm_start"] = ""
    if parsed.keep_first_last_fp32:
        config["keep_first_last_fp32"] = True
    config["strict_determinism"] = bool(parsed.strict_determinism or config.get("strict_determinism", False))
    config["smoke"] = parsed.smoke
    if config.get("kd_mode", "none") != "none":
        raise ValueError("Task 3 is no-KD only; kd_mode must be 'none'.")
    if config.get("keep_first_last_fp32", False) and config.get("condition") == "B2-default":
        raise ValueError("The official B2-default cannot leave first conv or FC in FP32.")
    required = {"run_name", "epochs", "batch_size", "num_workers", "data_root", "lr", "weight_decay", "warmup_epochs", "seeds"}
    missing = sorted(key for key in required if key not in config)
    if missing:
        raise ValueError(f"Task 3 config lacks required fields: {missing}")
    return argparse.Namespace(**config), config, config_path


def output_paths(run_name: str) -> dict[str, Path]:
    return {
        "checkpoint_dir": REPO_ROOT / "experiments" / "checkpoints" / run_name,
        "history_dir": REPO_ROOT / "experiments" / "histories" / run_name,
        "plot_dir": REPO_ROOT / "plots" / "task3" / run_name,
        "best_checkpoint": REPO_ROOT / "results" / "best_models" / f"{run_name}_best.pth",
        "summary": REPO_ROOT / "results" / f"{run_name}_summary.json",
        "diagnostics": REPO_ROOT / "results" / "task3_diagnostics" / run_name,
    }


def assert_fresh_outputs(paths: dict[str, Path], seeds: list[int]) -> None:
    reserved = [paths["best_checkpoint"], paths["summary"]]
    reserved.extend(paths["checkpoint_dir"] / f"resnet18_ternary_seed{seed}.pth" for seed in seeds)
    exists = [path for path in reserved if path.exists()]
    if exists:
        raise FileExistsError("Refusing to overwrite Task 3 evidence: " + ", ".join(map(str, exists)))


def build_ternary_student(cfg: QuantConfig, warm_start: str, keep_first_last_fp32: bool, device: torch.device):
    model = resnet18_cifar(num_classes=10)
    warmed = False
    warm_start_path = Path(warm_start) if warm_start else None
    if warm_start_path:
        if not warm_start_path.exists():
            raise FileNotFoundError(f"Requested Task 2 warm-start checkpoint does not exist: {warm_start_path}")
        checkpoint = torch.load(warm_start_path, map_location="cpu", weights_only=False)
        if checkpoint.get("arch") != "ResNet18-CIFAR" or checkpoint.get("num_classes") != 10:
            raise ValueError("Warm-start must be the frozen 10-class Task 2 ResNet18 checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        warmed = True
    convert_to_ternary(model, cfg, keep_first_last_fp32=keep_first_last_fp32)
    return model.to(device), warmed


def train_one_seed(seed: int, args: argparse.Namespace, cfg: QuantConfig, device: torch.device, paths: dict[str, Path], config_path: Path) -> dict:
    set_seed(seed, strict=args.strict_determinism)
    epochs = 1 if args.smoke else int(args.epochs)
    train_loader, val_loader, _, _ = get_train_val_loaders(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, loader_seed=seed,
    )
    model, warmed = build_ternary_student(cfg, args.warm_start, args.keep_first_last_fp32, device)
    conv_layers, linear_layers = count_ternary_layers(model)
    groups = ternary_parameter_groups(model, args.weight_decay)
    group_audit = parameter_group_audit(model, groups)
    if not group_audit["only_conv_linear_latents_decayed"] or group_audit["bn_bias_and_scale_tensors_decayed"]:
        raise AssertionError(f"Incorrect optimizer grouping: {group_audit}")
    optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, nesterov=bool(args.nesterov))
    scheduler = make_warmup_cosine(optimizer, int(args.warmup_epochs), epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    manifest = split_manifest(train_loader.dataset.dataset.targets)
    run_config = {
        "task": "T3", "condition": args.condition, "arch": "ResNet18-CIFAR-ternary", "seed": seed,
        "num_classes": 10, "epochs": epochs, "batch_size": args.batch_size, "effective_lr": args.lr,
        "optimizer": "SGD", "momentum": args.momentum, "nesterov": bool(args.nesterov),
        "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs, "scheduler": "warmup+cosine",
        "label_smoothing": args.label_smoothing, "strict_determinism": args.strict_determinism,
        "allow_tf32": False, "warm_started_from_fp32": warmed, "warm_start_path": args.warm_start or None,
        "keep_first_last_fp32": bool(args.keep_first_last_fp32), "ternary_conv_layers": conv_layers,
        "ternary_linear_layers": linear_layers, "normalization": {"mean": list(CIFAR10_MEAN), "std": list(CIFAR10_STD)},
        "split_manifest": manifest, "runtime": runtime_metadata(device), "config_source": str(config_path),
        "test_evaluation": "not_run", "kd_mode": "none", "parameter_group_audit": group_audit,
    }
    history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "gap")}
    best_val, best_epoch = -float("inf"), -1
    checkpoint_path = paths["checkpoint_dir"] / f"resnet18_ternary_seed{seed}.pth"
    started = time.perf_counter()
    print(f"\n=== {args.run_name} | seed {seed} | {epochs} epochs | {cfg.scope}/{cfg.threshold}/"
          f"{'sym' if cfg.symmetric else 'asym'}/{cfg.scale}/{cfg.ste} | warm-start={warmed} ===")
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            progress=lambda iterator: tqdm(iterator, desc=f"T3 {args.run_name} S{seed} {epoch}/{epochs}", leave=False),
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        for key, value in (("train_loss", train_loss), ("train_acc", train_acc), ("val_loss", val_loss),
                           ("val_acc", val_acc), ("lr", current_lr), ("gap", train_acc - val_acc)):
            history[key].append(value)
        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            raise FloatingPointError(f"Non-finite loss at seed={seed}, epoch={epoch}.")
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch, best_val, run_config,
                extra={"quant_config": cfg.serializable(), "keep_first_last_fp32": bool(args.keep_first_last_fp32)},
            )
        print(f"S{seed} E{epoch:3d}/{epochs} train={100*train_acc:5.2f}% val={100*val_acc:5.2f}% "
              f"loss={val_loss:.4f} lr={current_lr:.6f}" + ("  <-- best" if val_acc == best_val else ""))
    elapsed = time.perf_counter() - started
    reloaded, checkpoint = build_from_checkpoint(checkpoint_path, device)
    reload_loss, reload_acc = evaluate(reloaded, val_loader, criterion, device)
    reload_delta = abs(reload_acc - best_val)
    if reload_delta > 1 / 5000 + 1e-12:
        raise RuntimeError("Checkpoint reload differed by more than one validation example.")
    diagnostics = verify_model(reloaded, require_all_weight_layers=not args.keep_first_last_fp32, verbose=False)
    history_payload = {
        "history": history, "best_validation_accuracy": best_val, "best_epoch": best_epoch,
        "elapsed_seconds": elapsed, "seconds_per_epoch": elapsed / epochs, "config": run_config,
        "quant_config": cfg.serializable(), "checkpoint_reload_validation_accuracy": reload_acc,
        "checkpoint_reload_validation_loss": reload_loss, "checkpoint_reload_accuracy_delta": reload_delta,
        "quant_diagnostics": diagnostics,
    }
    history_path = paths["history_dir"] / f"resnet18_ternary_seed{seed}.json"
    save_json(history_path, history_payload)
    plot_paths = save_training_curves(history, paths["plot_dir"] / f"resnet18_ternary_seed{seed}")
    paths["diagnostics"].mkdir(parents=True, exist_ok=True)
    save_json(paths["diagnostics"] / f"seed{seed}.json", diagnostics)
    return {
        "seed": seed, "best_validation_accuracy": best_val, "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path), "history": str(history_path), "plots": plot_paths,
        "elapsed_seconds": elapsed, "seconds_per_epoch": elapsed / epochs,
        "final_augmented_train_accuracy": history["train_acc"][-1], "final_validation_accuracy": history["val_acc"][-1],
        "checkpoint_reload_validation_accuracy": reload_acc, "checkpoint_reload_accuracy_delta": reload_delta,
        "sparsity": diagnostics["sparsity"], "all_ternary": diagnostics["all_ternary"],
        "quant_error_mean": float(np.mean([row["quant_error"] for row in diagnostics["layers"]])),
    }


def main() -> None:
    args, config, config_path = parse_args()
    if args.keep_first_last_fp32:
        print("WARNING: diagnostic non-compliant run; it cannot be reported as B2-default.")
    cfg = QuantConfig(**args.quant).validate()
    paths = output_paths(args.run_name)
    assert_fresh_outputs(paths, list(args.seeds))
    device = get_device(require_cuda=True)
    print(f"Task 3 {args.run_name}: {torch.cuda.get_device_name(device)} | validation-only | test locked")
    print(f"Quant config: {cfg.serializable()}")
    results = [train_one_seed(seed, args, cfg, device, paths, config_path) for seed in args.seeds]
    accuracies = np.asarray([row["best_validation_accuracy"] for row in results])
    best = max(results, key=lambda row: row["best_validation_accuracy"])
    paths["best_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["checkpoint"], paths["best_checkpoint"])
    summary = {
        "task": "T3", "condition": args.condition, "run_name": args.run_name,
        "architecture": "ResNet18-CIFAR-ternary", "quant_config": cfg.serializable(),
        "all_conv_and_fc_ternary": not args.keep_first_last_fp32, "warm_start": args.warm_start or None,
        "training_seeds": list(args.seeds), "epochs": 1 if args.smoke else args.epochs,
        "validation_mean": float(accuracies.mean()),
        "validation_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
        "validation_best": float(accuracies.max()), "best_seed": best["seed"],
        "best_checkpoint": str(paths["best_checkpoint"]), "per_seed": results,
        "sparsity_mean": float(np.mean([row["sparsity"] for row in results])),
        "test_evaluation": "not_run", "test_mean": None, "test_std": None,
        "note": "Validation-only QAT result. No test-loader import or test evaluation occurred.",
    }
    save_json(paths["summary"], summary)
    print(f"Validation: {100*summary['validation_mean']:.2f}% ± {100*summary['validation_std']:.2f}%")
    print(f"Best checkpoint: {paths['best_checkpoint']}\nSummary: {paths['summary']}")


if __name__ == "__main__":
    # Task 3 remains byte-for-byte on its no-KD path.  The explicit Task 4
    # mode uses a separate trainer so it cannot silently alter the baseline.
    kd_mode = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--kd-mode=")), None)
    if kd_mode is None and "--kd-mode" in sys.argv:
        position = sys.argv.index("--kd-mode")
        if position + 1 < len(sys.argv):
            kd_mode = sys.argv[position + 1]
    if kd_mode == "vanilla":
        from scripts.train_student_vanilla_kd import main as task4_main
        task4_main()
    else:
        main()
