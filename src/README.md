# Source layout

This directory is intentionally task-gated by `KD.md`.

- `data.py` and `models/resnet_cifar.py` are created and frozen when **Task 2** is invoked.
- `quant/`, `kd/`, and `eval/` are created only by their corresponding tasks.
- No student, ternary quantizer, KD loss, or evaluation code belongs here until the preceding baseline task is complete.

Keeping this boundary prevents an unvalidated later-stage implementation from contaminating the immutable Task 1 baseline.
