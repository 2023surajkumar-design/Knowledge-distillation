# Source layout

This directory contains the completed production implementations for Tasks 2-4 and remains task-gated for future research extensions.

- `data.py` and `models/resnet_cifar.py` implement the frozen CIFAR-10 split and CIFAR ResNet baselines.
- `quant/` implements strict ternary QAT with latent/deployed separation and STE.
- `kd/` implements the frozen teacher loader and vanilla/DKD/DIST losses.
- `evaluation/verify_ternary.py` independently verifies deployed Conv/Linear weights.

Keeping this boundary prevents an unvalidated later-stage implementation from contaminating the immutable Task 1 baseline.
