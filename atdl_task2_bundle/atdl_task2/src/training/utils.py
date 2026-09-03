"""
src/training/utils.py  —  Shared training utilities (seeding, LR schedule,
train/eval loops, checkpointing, plotting). Used by all training scripts.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, strict: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except Exception as e:  # pragma: no cover
            print("Could not enable full determinism:", e)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup + cosine decay (identical recipe to Task 1)
# --------------------------------------------------------------------------- #
def make_warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device,
                    progress=None) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    iterator = progress(loader) if progress is not None else loader
    for inputs, labels in iterator:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             return_preds: bool = False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        if return_preds:
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
    loss_val, acc_val = running_loss / total, correct / total
    if return_preds:
        return loss_val, acc_val, torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()
    return loss_val, acc_val


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int,
                    best_acc: float, config: dict, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_accuracy": best_acc,
        "config": config,
        "seed": config.get("seed"),
        "arch": config.get("arch", "ResNet18-CIFAR"),
        "num_classes": config.get("num_classes", 10),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_history(path: str, history: dict, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"history": history, **meta}, f, indent=2)


# --------------------------------------------------------------------------- #
# Plotting (saved to disk — no interactive backend needed)
# --------------------------------------------------------------------------- #
def save_training_curves(history: dict, out_prefix: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    saved = []

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p = f"{out_prefix}_loss.png"; fig.savefig(p, dpi=120); plt.close(fig); saved.append(p)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [100 * a for a in history["train_acc"]], label="Train acc")
    plt.plot(epochs, [100 * a for a in history["val_acc"]], label="Val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p = f"{out_prefix}_acc.png"; fig.savefig(p, dpi=120); plt.close(fig); saved.append(p)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [100 * g for g in history["gap"]], color="tab:red")
    plt.axhline(0, color="k", lw=0.8, alpha=0.5)
    plt.xlabel("Epoch"); plt.ylabel("Train-Val gap (pp)"); plt.title("Generalization gap")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    p = f"{out_prefix}_gap.png"; fig.savefig(p, dpi=120); plt.close(fig); saved.append(p)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["lr"], color="tab:green")
    plt.xlabel("Epoch"); plt.ylabel("Learning rate"); plt.title("LR schedule")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    p = f"{out_prefix}_lr.png"; fig.savefig(p, dpi=120); plt.close(fig); saved.append(p)

    return saved
