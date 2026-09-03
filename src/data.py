"""Frozen CIFAR-10 data pipeline and test-set firewall for every ATDL task."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any, List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.1
DATA_ROOT_DEFAULT = "./data"

# These hashes fingerprint the literal Task 1 cell-14 algorithm on the official
# torchvision CIFAR-10 training targets. They are guards, not alternative split logic.
TASK1_TRAIN_INDEX_SHA256 = "c56f6a25318c9c65f072faff8490724c9054950f5af0d967daa4160b93ffd029"
TASK1_VAL_INDEX_SHA256 = "ffce04072b4dc04be36f12a4782477053bc3d8d9a60908013318934dbea9568c"


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return Task-1-identical train and deterministic evaluation transforms."""
    return (
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]),
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]),
    )


def _index_hash(indices: List[int]) -> str:
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()


def build_stratified_indices(
    targets: Any,
    val_fraction: float = VALIDATION_FRACTION,
    num_classes: int = NUM_CLASSES,
    seed: int = SPLIT_SEED,
) -> Tuple[List[int], List[int]]:
    """Reproduce Task 1's class-by-class seeded 45k/5k split exactly."""
    targets = np.asarray(targets)
    total = len(targets)
    if total != 50_000 or num_classes != NUM_CLASSES:
        raise ValueError("This frozen pipeline expects the 50,000-example, 10-class CIFAR-10 train set.")
    val_per_class = int(total * val_fraction / num_classes)
    if val_per_class != 500:
        raise ValueError("Frozen Task 1 split requires exactly 500 validation examples per class.")
    class_counts = Counter(targets.tolist())
    if {int(k): int(v) for k, v in class_counts.items()} != {i: 5_000 for i in range(NUM_CLASSES)}:
        raise ValueError("Unexpected CIFAR-10 class counts; refusing to construct a non-comparable split.")

    generator = torch.Generator().manual_seed(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for class_idx in range(num_classes):
        class_indices = np.flatnonzero(targets == class_idx)
        permuted = class_indices[torch.randperm(len(class_indices), generator=generator).numpy()]
        val_indices.extend(permuted[:val_per_class].tolist())
        train_indices.extend(permuted[val_per_class:].tolist())

    if len(train_indices) != 45_000 or len(val_indices) != 5_000:
        raise AssertionError("Frozen split size invariant failed.")
    if not set(train_indices).isdisjoint(val_indices):
        raise AssertionError("Train/validation overlap detected.")
    if seed == SPLIT_SEED and (
        _index_hash(train_indices) != TASK1_TRAIN_INDEX_SHA256
        or _index_hash(val_indices) != TASK1_VAL_INDEX_SHA256
    ):
        raise RuntimeError("Split fingerprint differs from Task 1; refusing an incomparable experiment.")
    return train_indices, val_indices


def split_manifest(targets: Any) -> dict:
    """Return the immutable split identity recorded in every experiment artifact."""
    train_indices, val_indices = build_stratified_indices(targets)
    target_array = np.asarray(targets)
    return {
        "dataset": "CIFAR-10",
        "split_algorithm": "task1_classwise_torch_generator_v1",
        "split_seed": SPLIT_SEED,
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "validation_per_class": {str(i): int((target_array[val_indices] == i).sum()) for i in range(NUM_CLASSES)},
        "train_index_sha256": _index_hash(train_indices),
        "validation_index_sha256": _index_hash(val_indices),
    }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_common(num_workers: int, seed: int) -> dict:
    return {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }


def get_train_val_loaders(
    data_root: str = DATA_ROOT_DEFAULT,
    batch_size: int = 128,
    num_workers: int = 4,
    loader_seed: int = SPLIT_SEED,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """Return augmented train and clean validation loaders; never the official test set."""
    train_transform, eval_transform = build_transforms()
    train_base = torchvision.datasets.CIFAR10(data_root, train=True, download=download, transform=train_transform)
    val_base = torchvision.datasets.CIFAR10(data_root, train=True, download=download, transform=eval_transform)
    train_indices, val_indices = build_stratified_indices(train_base.targets)
    # Independent generators prevent evaluation worker creation from perturbing train ordering.
    train_loader = DataLoader(
        Subset(train_base, train_indices), batch_size=batch_size, shuffle=True, drop_last=False,
        **_loader_common(num_workers, loader_seed),
    )
    val_loader = DataLoader(
        Subset(val_base, val_indices), batch_size=batch_size, shuffle=False, drop_last=False,
        **_loader_common(num_workers, loader_seed + 1),
    )
    return train_loader, val_loader, train_indices, val_indices


def get_test_loader(
    data_root: str = DATA_ROOT_DEFAULT,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    final_evaluation: bool = False,
) -> DataLoader:
    """Return the official test loader only after an explicit final-report unlock."""
    if not final_evaluation:
        raise RuntimeError(
            "TEST SET LOCKED (C9/C11). Use validation for every development decision; "
            "pass final_evaluation=True only for frozen final reporting."
        )
    _, eval_transform = build_transforms()
    test_dataset = torchvision.datasets.CIFAR10(
        data_root, train=False, download=download, transform=eval_transform
    )
    if len(test_dataset) != 10_000:
        raise ValueError("Unexpected official CIFAR-10 test-set size.")
    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        **_loader_common(num_workers, SPLIT_SEED),
    )
