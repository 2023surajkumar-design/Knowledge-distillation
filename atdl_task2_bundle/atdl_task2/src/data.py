"""
src/data.py  —  FROZEN shared CIFAR-10 data pipeline for the whole project.

This module is the single source of truth for:
  * CIFAR-10 loading + normalization constants
  * the fixed, class-stratified 45,000 / 5,000 train/validation split (seed 42)
  * the DataLoaders used by EVERY task (teacher, FP32 student, ternary student, KD)
  * the TEST-SET FIREWALL (C9 / C11 / Phase 42)

DO NOT MODIFY the split logic or the transforms after Task 1/2 — all cross-task
comparisons depend on an identical split and evaluation protocol (experimental
discipline in KD.md). The split below reproduces Task 1's split EXACTLY
(same seed, same per-class ordering), so the teacher's validation-based model
selection and every student experiment share the same held-out 5k validation
set and the same untouched 10k official test set.

Constraints honored here:
  C1  dataset = CIFAR-10 only
  C8  no external data
  C9  test set never used for selection/tuning/search
  C10 fixed stratified 45k / 5k / 10k split
  C11 test set evaluated only for final reporting (behind an explicit flag)
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# --------------------------------------------------------------------------- #
# Fixed constants (do not change — comparability across tasks)
# --------------------------------------------------------------------------- #
CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = 10

# Standard, widely-used CIFAR-10 channel statistics (identical to Task 1).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Fixed split parameters (identical to Task 1).
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.1          # -> 500 val images per class -> 5,000 val / 45,000 train
DATA_ROOT_DEFAULT = "./data"


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform).

    Train: RandomCrop(32, pad=4) + HorizontalFlip + ToTensor + Normalize.
    Eval : ToTensor + Normalize only (no random augmentation) — used for BOTH
           the validation split and the official test set.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


# --------------------------------------------------------------------------- #
# The frozen stratified split  (pure function — unit-testable without download)
# --------------------------------------------------------------------------- #
def build_stratified_indices(
    targets,
    val_fraction: float = VALIDATION_FRACTION,
    num_classes: int = NUM_CLASSES,
    seed: int = SPLIT_SEED,
) -> Tuple[List[int], List[int]]:
    """Deterministic, class-balanced train/val index split.

    Reproduces Task 1 EXACTLY: a single torch.Generator(seed) is consumed
    class-by-class in ascending class order; `val_per_class` items per class go
    to validation, the rest to train. Do not reorder these operations.
    """
    targets = np.asarray(targets)
    total = len(targets)
    val_per_class = int(total * val_fraction / num_classes)
    if val_per_class <= 0:
        raise ValueError("val_fraction too small to allocate one val sample per class.")

    gen = torch.Generator().manual_seed(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for class_idx in range(num_classes):
        class_indices = np.flatnonzero(targets == class_idx)
        perm = class_indices[torch.randperm(len(class_indices), generator=gen).numpy()]
        val_indices.extend(perm[:val_per_class].tolist())
        train_indices.extend(perm[val_per_class:].tolist())

    # Integrity guarantees (asserted at every call).
    assert len(train_indices) + len(val_indices) == total
    assert set(train_indices).isdisjoint(val_indices)
    return train_indices, val_indices


# --------------------------------------------------------------------------- #
# Reproducible DataLoader worker seeding
# --------------------------------------------------------------------------- #
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_common(num_workers: int, seed: int) -> dict:
    use_pin = torch.cuda.is_available()
    persistent = num_workers > 0
    gen = torch.Generator().manual_seed(seed)
    return dict(
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent,
        worker_init_fn=seed_worker,
        generator=gen,
    )


# --------------------------------------------------------------------------- #
# Public loader API
# --------------------------------------------------------------------------- #
def get_train_val_loaders(
    data_root: str = DATA_ROOT_DEFAULT,
    batch_size: int = 128,
    num_workers: int = 4,
    loader_seed: int = SPLIT_SEED,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """Return (train_loader, val_loader, train_indices, val_indices).

    train_loader uses augmentation; val_loader uses the deterministic eval
    transform. Both are Subsets of the official CIFAR-10 *training* set under
    the frozen stratified split. The official test set is NOT returned here.
    """
    train_transform, eval_transform = build_transforms()

    train_base = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download, transform=train_transform)
    val_base = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download, transform=eval_transform)

    train_idx, val_idx = build_stratified_indices(train_base.targets)

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)

    common = _loader_common(num_workers, loader_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            drop_last=False, **common)
    return train_loader, val_loader, train_idx, val_idx


def get_test_loader(
    data_root: str = DATA_ROOT_DEFAULT,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    final_evaluation: bool = False,
) -> DataLoader:
    """TEST-SET FIREWALL (C9 / C11 / Phase 42).

    The official 10,000-image test set may ONLY be used for final reporting.
    This function refuses to return the test loader unless the caller explicitly
    sets `final_evaluation=True`. Never call it during training, HPO, Optuna
    scoring, AutoResearch scoring, ablation search, or development.
    """
    if not final_evaluation:
        raise RuntimeError(
            "TEST SET LOCKED (C9/C11). The official test set is for FINAL reporting only. "
            "Use get_train_val_loaders() and select models on the validation split. "
            "If (and only if) you are producing the final report, pass final_evaluation=True."
        )
    _, eval_transform = build_transforms()
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=download, transform=eval_transform)
    common = _loader_common(num_workers, SPLIT_SEED)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      drop_last=False, **common)
