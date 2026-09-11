#!/usr/bin/env python3
"""Task 4 production trainer: vanilla logit KD with strict ternary QAT.

This is purposefully separate from the immutable Task 3 implementation.  The
public ``train_student_ternary.py --kd-mode vanilla`` entry point dispatches
here; ``--kd-mode none`` continues using the exact Task 3 path.
"""

from __future__ import annotations

import argparse
import math
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
from src.kd.losses import decoupled_kd_loss, dist_loss, vanilla_kd_loss
from src.kd.teacher import assert_teacher_frozen, load_frozen_teacher
from src.models.resnet_cifar import resnet18_cifar
from src.quant.ternary import QuantConfig, TernaryConv2d, TernaryLinear, convert_to_ternary, parameter_group_audit, ternary_parameter_groups
from src.training.utils import evaluate, get_device, make_warmup_cosine, runtime_metadata, save_checkpoint, save_json, save_training_curves, set_seed


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Task 4 vanilla logit-KD + strict ternary QAT.")
    p.add_argument("--kd-mode", choices=["vanilla", "dkd", "dist"], default="vanilla")
    p.add_argument("--dkd-alpha", type=float); p.add_argument("--dkd-beta", type=float)
    p.add_argument("--config", default="configs/kd/resnet18_ternary_vanillaKD.yaml")
    p.add_argument("--run-name"); p.add_argument("--condition")
    p.add_argument("--epochs", type=int); p.add_argument("--batch-size", type=int); p.add_argument("--num-workers", type=int)
    p.add_argument("--data-root"); p.add_argument("--lr", type=float); p.add_argument("--weight-decay", type=float)
    p.add_argument("--momentum", type=float); p.add_argument("--warmup-epochs", type=int); p.add_argument("--label-smoothing", type=float)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--teacher-checkpoint"); p.add_argument("--teacher-tag")
    p.add_argument("--temperature", type=float); p.add_argument("--kd-lambda", type=float)
    p.add_argument("--warm-start"); p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--gradient-diagnostics", action="store_true", default=None)
    p.add_argument("--no-gradient-diagnostics", dest="gradient_diagnostics", action="store_false")
    p.add_argument("--smoke", action="store_true"); p.add_argument("--strict-determinism", action="store_true")
    return p


def parse_args() -> tuple[argparse.Namespace, Path]:
    cli = parser().parse_args()
    path = Path(cli.config)
    config = yaml.safe_load(path.read_text())
    overrides = {
        "run_name": cli.run_name, "condition": cli.condition, "epochs": cli.epochs, "batch_size": cli.batch_size,
        "num_workers": cli.num_workers, "data_root": cli.data_root, "lr": cli.lr, "weight_decay": cli.weight_decay,
        "momentum": cli.momentum, "warmup_epochs": cli.warmup_epochs, "label_smoothing": cli.label_smoothing,
        "seeds": cli.seeds, "teacher_checkpoint": cli.teacher_checkpoint, "teacher_tag": cli.teacher_tag,
        "temperature": cli.temperature, "kd_lambda": cli.kd_lambda, "warm_start": cli.warm_start,
        "gradient_diagnostics": cli.gradient_diagnostics, "dkd_alpha": cli.dkd_alpha, "dkd_beta": cli.dkd_beta,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["kd_mode"] = cli.kd_mode
    if cli.no_warm_start:
        config["warm_start"] = ""
    config["smoke"] = cli.smoke
    config["strict_determinism"] = bool(cli.strict_determinism or config.get("strict_determinism", False))
    required = {"condition", "run_name", "data_root", "epochs", "batch_size", "num_workers", "lr", "weight_decay", "momentum", "warmup_epochs", "label_smoothing", "seeds", "teacher_checkpoint", "temperature", "kd_lambda", "quant"}
    missing = sorted(key for key in required if key not in config)
    if missing:
        raise ValueError(f"Task 4 configuration missing {missing}.")
    if config.get("keep_first_last_fp32", False):
        raise ValueError("Task 4 official runs must ternarize first convolution and final FC.")
    if not config["warm_start"] and config["condition"] == "B3-vanillaKD-canonical":
        raise ValueError("The canonical Task 4 condition requires the frozen Task 2 warm-start; random init is a labelled diagnostic only.")
    return argparse.Namespace(**config), path


def output_paths(run_name: str) -> dict[str, Path]:
    root = REPO_ROOT / "experiments" / "task4"
    return {
        "checkpoint_dir": root / "checkpoints" / run_name,
        "history_dir": root / "histories" / run_name,
        "plot_dir": REPO_ROOT / "plots" / "task4" / run_name,
        "diagnostic_dir": root / "diagnostics" / run_name,
        "best_checkpoint": REPO_ROOT / "results" / "task4" / "best_models" / f"{run_name}_best.pth",
        "summary": REPO_ROOT / "results" / "task4" / f"{run_name}_summary.json",
    }


def assert_fresh(paths: dict[str, Path], seeds: list[int]) -> None:
    candidates = [paths["summary"], paths["best_checkpoint"]]
    candidates += [paths["checkpoint_dir"] / f"resnet18_ternary_kd_seed{seed}.pth" for seed in seeds]
    existing = [str(path) for path in candidates if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite Task 4 evidence: " + ", ".join(existing))


def build_student(cfg: QuantConfig, warm_start: str, device: torch.device) -> torch.nn.Module:
    """Construct the student, accepting random initialization only for labelled diagnostics."""
    if not warm_start:
        model = resnet18_cifar(num_classes=10)
        convert_to_ternary(model, cfg, keep_first_last_fp32=False)
        return model.to(device)
    checkpoint_path = Path(warm_start)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Task 4 warm-start absent: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("arch") != "ResNet18-CIFAR" or checkpoint.get("num_classes") != 10:
        raise ValueError("Task 4 warm-start must be the frozen Task 2 FP32 ResNet18 checkpoint.")
    model = resnet18_cifar(num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    convert_to_ternary(model, cfg, keep_first_last_fp32=False)
    return model.to(device)


def _latent_weights(student: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [module.weight for module in student.modules() if isinstance(module, (TernaryConv2d, TernaryLinear))]


def gradient_diagnostics(ce: torch.Tensor, kd: torch.Tensor, parameters: list[torch.nn.Parameter]) -> dict:
    ce_grads = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
    kd_grads = torch.autograd.grad(kd, parameters, retain_graph=True, allow_unused=True)
    ce_sq = kd_sq = dot = 0.0
    for ce_grad, kd_grad in zip(ce_grads, kd_grads):
        if ce_grad is None or kd_grad is None:
            continue
        ce_sq += float(ce_grad.detach().float().square().sum().item())
        kd_sq += float(kd_grad.detach().float().square().sum().item())
        dot += float((ce_grad.detach().float() * kd_grad.detach().float()).sum().item())
    ce_norm, kd_norm = math.sqrt(ce_sq), math.sqrt(kd_sq)
    return {"ce_grad_norm": ce_norm, "kd_grad_norm": kd_norm, "grad_cosine": dot / (ce_norm * kd_norm + 1e-12)}


def _distill(mode, student_logits, teacher_logits, labels, temperature, kd_lambda, criterion, alpha, beta):
    if mode == "dkd":
        return decoupled_kd_loss(student_logits, teacher_logits, labels, temperature, kd_lambda, criterion, alpha, beta)
    if mode == "dist":
        return dist_loss(student_logits, teacher_logits, labels, temperature, kd_lambda, criterion, alpha, beta)
    return vanilla_kd_loss(student_logits, teacher_logits, labels, temperature, kd_lambda, criterion)


def train_epoch(student, teacher, loader, optimizer, device, temperature, kd_lambda, criterion, with_gradient_diagnostics: bool, description: str, mode="vanilla", alpha=1., beta=8.) -> dict:
    student.train(); teacher.eval(); assert_teacher_frozen(teacher)
    totals = {key: 0.0 for key in ("total_loss", "ce_loss", "kd_loss", "weighted_ce", "weighted_kd", "correct")}
    gradient_rows = []
    latent = _latent_weights(student)
    for batch_index, (inputs, labels) in enumerate(tqdm(loader, desc=description, leave=False)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        optimizer.zero_grad(set_to_none=True)
        student_logits = student(inputs)
        losses = _distill(mode, student_logits, teacher_logits, labels, temperature, kd_lambda, criterion, alpha, beta)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("Non-finite Task 4 loss.")
        if with_gradient_diagnostics and batch_index == 0:
            gradient_rows.append(gradient_diagnostics(losses.ce, losses.kd, latent))
        losses.total.backward()
        assert_teacher_frozen(teacher, assert_no_gradients=True)
        optimizer.step()
        batch_size = labels.size(0)
        totals["total_loss"] += float(losses.total.detach().item()) * batch_size
        totals["ce_loss"] += float(losses.ce.detach().item()) * batch_size
        totals["kd_loss"] += float(losses.kd.detach().item()) * batch_size
        totals["weighted_ce"] += float(losses.weighted_ce.detach().item()) * batch_size
        totals["weighted_kd"] += float(losses.weighted_kd.detach().item()) * batch_size
        totals["correct"] += student_logits.argmax(1).eq(labels).sum().item()
        totals.setdefault("samples", 0); totals["samples"] += batch_size
    count = totals.pop("samples")
    result = {key: value / count for key, value in totals.items() if key != "correct"}
    result["accuracy"] = totals["correct"] / count
    if gradient_rows:
        result.update({key: float(np.mean([row[key] for row in gradient_rows])) for key in gradient_rows[0]})
    return result


@torch.no_grad()
def validation_diagnostics(student, teacher, loader, criterion, device, temperature, kd_lambda, mode="vanilla", alpha=1., beta=8.) -> dict:
    student.eval(); teacher.eval(); assert_teacher_frozen(teacher)
    sums = {key: 0.0 for key in ("ce", "kd", "teacher_correct", "student_correct", "agreement", "teacher_entropy", "student_entropy", "teacher_confidence", "student_confidence", "teacher_margin", "student_margin")}
    count = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        teacher_logits, student_logits = teacher(inputs), student(inputs)
        losses = _distill(mode, student_logits, teacher_logits, labels, temperature, kd_lambda, criterion, alpha, beta)
        teacher_prob = torch.softmax(teacher_logits, dim=1); student_prob = torch.softmax(student_logits, dim=1)
        teacher_top2 = teacher_prob.topk(2, dim=1).values; student_top2 = student_prob.topk(2, dim=1).values
        batch_size = labels.size(0); count += batch_size
        sums["ce"] += float(losses.ce.item()) * batch_size; sums["kd"] += float(losses.kd.item()) * batch_size
        sums["teacher_correct"] += teacher_logits.argmax(1).eq(labels).sum().item(); sums["student_correct"] += student_logits.argmax(1).eq(labels).sum().item()
        sums["agreement"] += teacher_logits.argmax(1).eq(student_logits.argmax(1)).sum().item()
        sums["teacher_entropy"] += float((-(teacher_prob * teacher_prob.clamp_min(1e-12).log()).sum(1)).sum().item())
        sums["student_entropy"] += float((-(student_prob * student_prob.clamp_min(1e-12).log()).sum(1)).sum().item())
        sums["teacher_confidence"] += float(teacher_prob.max(1).values.sum().item()); sums["student_confidence"] += float(student_prob.max(1).values.sum().item())
        sums["teacher_margin"] += float((teacher_top2[:, 0] - teacher_top2[:, 1]).sum().item()); sums["student_margin"] += float((student_top2[:, 0] - student_top2[:, 1]).sum().item())
    return {key: value / count for key, value in sums.items()}


def save_kd_plots(history: dict, prefix: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prefix.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["total_loss"]) + 1)
    output = []
    for name, series, ylabel in [
        ("contributions", [(history["weighted_ce"], "(1−λ) CE"), (history["weighted_kd"], "λ T² KL")], "Weighted loss"),
        ("gradient", [(history["ce_grad_norm"], "CE gradient norm"), (history["kd_grad_norm"], "KD gradient norm")], "Norm"),
        ("gradient_cosine", [(history["grad_cosine"], "cos(∇CE, ∇KD)")], "Cosine"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for values, label in series:
            ax.plot(epochs, values, label=label)
        if name == "gradient_cosine": ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Epoch", ylabel=ylabel); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        path = f"{prefix}_{name}.png"; fig.savefig(path, dpi=120); plt.close(fig); output.append(path)
    return output


def train_one_seed(seed: int, args: argparse.Namespace, cfg: QuantConfig, teacher, teacher_metadata: dict, device: torch.device, paths: dict, config_path: Path) -> dict:
    set_seed(seed, strict=args.strict_determinism)
    epochs = 1 if args.smoke else int(args.epochs)
    train_loader, validation_loader, _, _ = get_train_val_loaders(args.data_root, args.batch_size, args.num_workers, loader_seed=seed)
    student = build_student(cfg, args.warm_start, device)
    groups = ternary_parameter_groups(student, args.weight_decay)
    grouping = parameter_group_audit(student, groups)
    if not grouping["only_conv_linear_latents_decayed"] or grouping["bn_bias_and_scale_tensors_decayed"]:
        raise AssertionError(f"Bad Task 4 parameter grouping: {grouping}")
    optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    scheduler = make_warmup_cosine(optimizer, args.warmup_epochs, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    manifest = split_manifest(train_loader.dataset.dataset.targets)
    run_config = {
        "task": "T4", "condition": args.condition, "arch": "ResNet18-CIFAR-ternary", "seed": seed, "num_classes": 10,
        "epochs": epochs, "batch_size": args.batch_size, "lr": args.lr, "optimizer": "SGD", "momentum": args.momentum,
        "nesterov": args.nesterov, "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
        "label_smoothing": args.label_smoothing, "scheduler": "warmup+cosine", "quant_config": cfg.serializable(),
        "teacher": teacher_metadata, "temperature": args.temperature, "kd_lambda": args.kd_lambda,
        "warm_start": args.warm_start, "keep_first_last_fp32": False, "kd_mode": args.kd_mode, "dkd_alpha": args.dkd_alpha, "dkd_beta": args.dkd_beta,
        "split_manifest": manifest, "normalization": {"mean": list(CIFAR10_MEAN), "std": list(CIFAR10_STD)},
        "runtime": runtime_metadata(device), "config_source": str(config_path), "parameter_group_audit": grouping,
        "test_evaluation": "not_run",
    }
    history = {key: [] for key in ("total_loss", "ce_loss", "kd_loss", "weighted_ce", "weighted_kd", "train_acc", "val_loss", "val_acc", "lr", "gap", "ce_grad_norm", "kd_grad_norm", "grad_cosine")}
    best_val, best_epoch = -float("inf"), -1
    checkpoint_path = paths["checkpoint_dir"] / f"resnet18_ternary_kd_seed{seed}.pth"
    started = time.perf_counter()
    print(f"\n=== {args.run_name} seed {seed} | T={args.temperature:g}, lambda={args.kd_lambda:g} | {epochs} epochs ===")
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train = train_epoch(student, teacher, train_loader, optimizer, device, args.temperature, args.kd_lambda, criterion, bool(args.gradient_diagnostics), f"T4 S{seed} {epoch}/{epochs}", args.kd_mode, args.dkd_alpha, args.dkd_beta)
        val_loss, val_acc = evaluate(student, validation_loader, criterion, device)
        scheduler.step()
        for key in ("total_loss", "ce_loss", "kd_loss", "weighted_ce", "weighted_kd"):
            history[key].append(train[key])
        history["train_acc"].append(train["accuracy"]); history["val_loss"].append(val_loss); history["val_acc"].append(val_acc); history["lr"].append(lr); history["gap"].append(train["accuracy"] - val_acc)
        for key in ("ce_grad_norm", "kd_grad_norm", "grad_cosine"):
            history[key].append(train.get(key, float("nan")))
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            save_checkpoint(checkpoint_path, student, optimizer, scheduler, epoch, best_val, run_config, extra={"quant_config": cfg.serializable(), "teacher_metadata": teacher_metadata, "keep_first_last_fp32": False})
        print(f"S{seed} E{epoch:3d}/{epochs} total={train['total_loss']:.4f} CE={train['ce_loss']:.4f} KD={train['kd_loss']:.4f} val={100*val_acc:.2f}% lr={lr:.6f}" + (" <-- best" if val_acc == best_val else ""))
    elapsed = time.perf_counter() - started
    reloaded, checkpoint = build_from_checkpoint(checkpoint_path, device)
    reload_loss, reload_acc = evaluate(reloaded, validation_loader, criterion, device)
    if abs(reload_acc - best_val) > 1 / 5000 + 1e-12:
        raise RuntimeError("Reloaded Task 4 checkpoint differs by more than one validation example.")
    diagnostics = verify_model(reloaded, require_all_weight_layers=True, verbose=False)
    val_kd = validation_diagnostics(reloaded, teacher, validation_loader, criterion, device, args.temperature, args.kd_lambda, args.kd_mode, args.dkd_alpha, args.dkd_beta)
    paths["history_dir"].mkdir(parents=True, exist_ok=True); paths["diagnostic_dir"].mkdir(parents=True, exist_ok=True)
    history_path = paths["history_dir"] / f"seed{seed}.json"
    save_json(history_path, {"history": history, "best_validation_accuracy": best_val, "best_epoch": best_epoch, "elapsed_seconds": elapsed, "config": run_config, "validation_kd_diagnostics": val_kd, "quant_diagnostics": diagnostics, "checkpoint_reload_validation_accuracy": reload_acc, "checkpoint_reload_validation_loss": reload_loss, "checkpoint_reload_accuracy_delta": abs(reload_acc-best_val)})
    plots = save_training_curves({"train_loss": history["total_loss"], "train_acc": history["train_acc"], "val_loss": history["val_loss"], "val_acc": history["val_acc"], "lr": history["lr"], "gap": history["gap"]}, paths["plot_dir"] / f"seed{seed}") + save_kd_plots(history, paths["plot_dir"] / f"seed{seed}")
    save_json(paths["diagnostic_dir"] / f"seed{seed}.json", {"ternary": diagnostics, "validation_kd": val_kd, "teacher_metadata": teacher_metadata})
    return {"seed": seed, "best_validation_accuracy": best_val, "best_epoch": best_epoch, "final_validation_accuracy": history["val_acc"][-1], "final_augmented_train_accuracy": history["train_acc"][-1], "checkpoint": str(checkpoint_path), "history": str(history_path), "plots": plots, "elapsed_seconds": elapsed, "seconds_per_epoch": elapsed/epochs, "sparsity": diagnostics["sparsity"], "quant_error_mean": float(np.mean([row["quant_error"] for row in diagnostics["layers"]])), "all_ternary": diagnostics["all_ternary"], "checkpoint_reload_accuracy_delta": abs(reload_acc-best_val), "validation_kd_diagnostics": val_kd}


def main() -> None:
    args, config_path = parse_args()
    args.dkd_alpha = float(getattr(args, "dkd_alpha", 1.0) if getattr(args, "dkd_alpha", None) is not None else 1.0)
    args.dkd_beta = float(getattr(args, "dkd_beta", 8.0) if getattr(args, "dkd_beta", None) is not None else 8.0)
    cfg = QuantConfig(**args.quant).validate()
    if args.temperature <= 0 or not 0 <= args.kd_lambda <= 1:
        raise ValueError("temperature must be positive and kd_lambda must lie in [0, 1].")
    paths = output_paths(args.run_name); assert_fresh(paths, list(args.seeds))
    device = get_device(require_cuda=True)
    teacher, _, teacher_metadata = load_frozen_teacher(args.teacher_checkpoint, device)
    print(f"Task 4 validation-only: teacher={teacher_metadata['path']} SHA256={teacher_metadata['sha256'][:12]}…")
    results = [train_one_seed(seed, args, cfg, teacher, teacher_metadata, device, paths, config_path) for seed in args.seeds]
    scores = np.asarray([row["best_validation_accuracy"] for row in results]); best = max(results, key=lambda row: row["best_validation_accuracy"])
    paths["best_checkpoint"].parent.mkdir(parents=True, exist_ok=True); shutil.copy2(best["checkpoint"], paths["best_checkpoint"])
    summary = {"task": "T4", "condition": args.condition, "run_name": args.run_name, "architecture": "ResNet18-CIFAR-ternary", "kd_mode": args.kd_mode, "temperature": args.temperature, "kd_lambda": args.kd_lambda, "dkd_alpha": args.dkd_alpha, "dkd_beta": args.dkd_beta, "teacher": teacher_metadata, "quant_config": cfg.serializable(), "warm_start": args.warm_start, "all_conv_and_fc_ternary": True, "training_seeds": list(args.seeds), "epochs": 1 if args.smoke else args.epochs, "validation_mean": float(scores.mean()), "validation_std": float(scores.std(ddof=1)) if len(scores)>1 else 0.0, "validation_best": float(scores.max()), "best_seed": best["seed"], "best_checkpoint": str(paths["best_checkpoint"]), "per_seed": results, "sparsity_mean": float(np.mean([row["sparsity"] for row in results])), "test_evaluation": "not_run", "test_mean": None, "test_std": None, "note": "Validation-only strict ternary QAT; no test loader is imported."}
    save_json(paths["summary"], summary)
    print(f"Task 4 validation: {100*summary['validation_mean']:.2f}% ± {100*summary['validation_std']:.2f}%\nSummary: {paths['summary']}")


if __name__ == "__main__":
    main()
