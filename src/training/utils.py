"""Shared reproducible FP32 training, evaluation, checkpointing, and plotting utilities."""

from __future__ import annotations

import json
import math
import os
import platform
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def configure_fp32_cuda(allow_tf32: bool = False) -> None:
    """Match the Task 1 IEEE-FP32 math policy on Ampere-or-newer CUDA GPUs."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if not allow_tf32:
            torch.set_float32_matmul_precision("highest")


def set_seed(seed: int, strict: bool = False) -> None:
    """Seed Python, NumPy, Torch, CUDA, and worker-derived RNG streams."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if strict:
        # Must be set before the first CUDA matmul that needs deterministic workspace.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    configure_fp32_cuda(allow_tf32=False)


def get_device(require_cuda: bool = True) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this controlled experiment; refusing an accidental CPU run.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runtime_metadata(device: torch.device) -> dict:
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "allow_tf32": False,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update({
            "gpu_name": properties.name,
            "gpu_vram_bytes": properties.total_memory,
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        })
    return metadata


def make_warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    """Task-1-identical LambdaLR: epochs 1..5 use .02,.04,.06,.08,.10 then cosine."""
    if total_epochs < 1 or warmup_epochs < 0:
        raise ValueError("total_epochs must be positive and warmup_epochs non-negative.")

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def preview_warmup_cosine(base_lr: float, warmup_epochs: int, total_epochs: int) -> list[float]:
    """Return learning rates actually used at the start of each optimizer epoch."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive.")
    def multiplier(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return [base_lr * multiplier(epoch) for epoch in range(total_epochs)]


def train_one_epoch(model, loader, criterion, optimizer, device, progress=None) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    iterator = progress(loader) if progress is not None else loader
    for inputs, labels in iterator:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss encountered.")
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise RuntimeError("Training loader produced zero samples.")
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds: bool = False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss encountered.")
        running_loss += loss.item() * inputs.size(0)
        predicted = logits.argmax(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        if return_preds:
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
    if total == 0:
        raise RuntimeError("Evaluation loader produced zero samples.")
    loss_value, accuracy = running_loss / total, correct / total
    if return_preds:
        return loss_value, accuracy, torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()
    return loss_value, accuracy


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_accuracy: float,
    config: dict,
    extra: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_accuracy": best_val_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "best_validation_accuracy": best_val_accuracy,
        "selection_metric": "validation_accuracy",
        "config": config,
        "seed": config["seed"],
        "arch": config["arch"],
        "num_classes": config["num_classes"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dataset": "CIFAR-10",
        "split": config["split_manifest"],
        "runtime": config["runtime"],
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_training_curves(history: dict, out_prefix: str | Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_prefix = str(out_prefix)
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    outputs = []
    for suffix, series, ylabel, title in [
        ("loss", [(history["train_loss"], "Train loss"), (history["val_loss"], "Validation loss")], "Loss", "Loss"),
        ("acc", [([100 * x for x in history["train_acc"]], "Augmented-train accuracy"), ([100 * x for x in history["val_acc"]], "Validation accuracy")], "Accuracy (%)", "Accuracy"),
        ("gap", [([100 * x for x in history["gap"]], "Augmented-train minus validation")], "Difference (pp)", "Augmented-training accuracy gap"),
        ("lr", [(history["lr"], "Learning rate")], "Learning rate", "Learning-rate schedule"),
    ]:
        fig = plt.figure(figsize=(7, 4.5))
        for values, label in series:
            plt.plot(epochs, values, label=label)
        if suffix == "gap":
            plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        output = f"{out_prefix}_{suffix}.png"
        fig.savefig(output, dpi=120); plt.close(fig); outputs.append(output)
    return outputs
