#!/usr/bin/env bash
set -euo pipefail

# Create an isolated environment and register it as the notebook kernel.
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cuda.txt
.venv/bin/python -m ipykernel install --user \
  --name atdl-task1-cuda \
  --display-name "Python (ATDL Task 1 CUDA)"

.venv/bin/python - <<'PY'
import torch
import torchvision

print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check nvidia-smi and the NVIDIA driver before training.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
