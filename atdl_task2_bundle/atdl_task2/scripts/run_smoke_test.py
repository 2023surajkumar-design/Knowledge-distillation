#!/usr/bin/env python3
"""
scripts/run_smoke_test.py  —  Fast end-to-end pipeline check (STEP 12 in the plan).

Runs the Task 2 pipeline for ONE epoch on a small subset so you can confirm,
in ~1 minute, that data loading, the model, the training loop, checkpointing,
and plotting all work on your machine before launching the full 3-seed run.
This produces NO scientific result — it only validates the pipeline.

Usage:
    uv run scripts/run_smoke_test.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data import get_train_val_loaders, get_test_loader  # noqa: E402
from src.models.resnet_cifar import resnet18_cifar, resnet34_cifar, count_parameters  # noqa: E402
from src.training.utils import (  # noqa: E402
    set_seed, get_device, make_warmup_cosine, train_one_epoch, evaluate, save_checkpoint,
)


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Device: {device} | torch {torch.__version__}")

    # 1) Model sanity.
    s = resnet18_cifar(10).to(device)
    t = resnet34_cifar(10).to(device)
    s.eval(); t.eval()
    with torch.no_grad():
        dummy = torch.randn(4, 3, 32, 32, device=device)
        assert s(dummy).shape == (4, 10)
        assert t(dummy).shape == (4, 10)
    print(f"ResNet18 params: {count_parameters(s):,}  |  ResNet34 params: {count_parameters(t):,}")
    assert all(p.dtype == torch.float32 for p in s.parameters())

    # 2) Data + one tiny training epoch on a subset.
    train_loader, val_loader, tr_idx, va_idx = get_train_val_loaders(batch_size=128, num_workers=2)
    print(f"Split: {len(tr_idx)} train / {len(va_idx)} val")
    assert len(tr_idx) == 45000 and len(va_idx) == 5000

    small_train = DataLoader(Subset(train_loader.dataset, list(range(512))),
                             batch_size=128, shuffle=True, num_workers=0)
    small_val = DataLoader(Subset(val_loader.dataset, list(range(256))),
                           batch_size=128, shuffle=False, num_workers=0)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(s.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = make_warmup_cosine(opt, warmup_epochs=0, total_epochs=1)

    tr_loss, tr_acc = train_one_epoch(s, small_train, crit, opt, device)
    sched.step()
    va_loss, va_acc = evaluate(s, small_val, crit, device)
    print(f"1-epoch subset: train {tr_loss:.3f}/{100*tr_acc:.1f}% | val {va_loss:.3f}/{100*va_acc:.1f}%")

    # 3) Checkpoint round-trip.
    os.makedirs("experiments/checkpoints", exist_ok=True)
    p = "experiments/checkpoints/_smoke.pth"
    save_checkpoint(p, s, opt, sched, epoch=1, best_acc=va_acc, config={"seed": 42})
    s2 = resnet18_cifar(10).to(device)
    s2.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model_state_dict"])
    print("Checkpoint save/reload: OK")

    # 4) Firewall check.
    try:
        get_test_loader()  # must refuse
        raise AssertionError("Firewall FAILED: test loader returned without flag!")
    except RuntimeError:
        print("Test-set firewall: OK (locked without --final-evaluation)")

    print("\nSMOKE TEST PASSED — pipeline is ready for the full run.")


if __name__ == "__main__":
    main()
