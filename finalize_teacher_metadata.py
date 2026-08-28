"""Align the completed inference checkpoint's selection metadata with the notebook protocol."""

from pathlib import Path

import torch


path = Path("resnet34_cifar10_fp32_final.pth")
checkpoint = torch.load(path, map_location="cpu")
checkpoint["best_accuracy"] = checkpoint["best_validation_accuracy"]
checkpoint["selection_metric"] = "validation_accuracy"
torch.save(checkpoint, path)
print(f"Updated {path}: best_accuracy={checkpoint['best_accuracy']:.4f} (validation)")
