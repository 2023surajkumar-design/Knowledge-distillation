"""
src/models/resnet_cifar.py  —  Shared CIFAR-adapted ResNet definitions.

Both the ResNet34 teacher (Task 1) and the ResNet18 student (Task 2+) live here
so every task imports the SAME architecture code.

CIFAR adaptation (vs canonical ImageNet ResNet):
  * stem = 3x3 stride-1 conv, padding 1 (NOT 7x7 stride-2)
  * NO initial max-pool (preserves 32x32 resolution)
  * canonical block configs preserved:
        ResNet18 -> BasicBlock [2, 2, 2, 2]   (~11.17M params)
        ResNet34 -> BasicBlock [3, 4, 6, 3]   (~21.28M params)
  * stage channels 64/128/256/512; GAP -> FC(num_classes)

The ResNet34 definition here is byte-for-byte compatible with the Task 1
checkpoint (same module names: conv1, bn1, layer1..4, fc), so
`resnet34_cifar(num_classes=10).load_state_dict(ckpt["model_state_dict"])`
loads the frozen teacher without surgery.

NOTE (Task 2 scope): this file defines standard FP32 modules only. Ternary
layers live in src/quant/ (Task 3+) and are NOT introduced here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock: two 3x3 convs + identity/projection shortcut."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """CIFAR ResNet with a 3x3 stride-1 stem and no initial max-pool."""

    def __init__(self, block, num_blocks, num_classes: int = 10,
                 return_features: bool = False):
        super().__init__()
        self.in_channels = 64
        self.return_features = return_features

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        pooled = self.avgpool(f4)
        flat = torch.flatten(pooled, 1)
        logits = self.fc(flat)
        if self.return_features:
            # Stage feature maps + penultimate vector — used by feature/relation
            # KD in later tasks. Harmless (and unused) for the FP32 baseline.
            return logits, {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "penultimate": flat}
        return logits


def resnet18_cifar(num_classes: int = 10, return_features: bool = False) -> ResNetCIFAR:
    """CIFAR ResNet18 — BasicBlock [2, 2, 2, 2] (~11.17M params)."""
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                       return_features=return_features)


def resnet34_cifar(num_classes: int = 10, return_features: bool = False) -> ResNetCIFAR:
    """CIFAR ResNet34 — BasicBlock [3, 4, 6, 3] (~21.28M params). Task 1 teacher."""
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                       return_features=return_features)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
