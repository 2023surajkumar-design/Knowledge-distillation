# ATDL Task 1 — FP32 CIFAR-10 ResNet34 Teacher

The project now includes completed Task 1 through Task 4 implementations and preserved validation artifacts: FP32 ResNet34 teacher, FP32 ResNet18 baseline, strict ternary ResNet18 QAT without KD, and vanilla ternary KD. DKD and DIST are supplementary validation-only experiments. The official test set remains locked in the research phase.

## CUDA environment

The training environment is a Python 3.13 virtual environment in `.venv`, with CUDA-enabled PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128. The notebook requires CUDA and fails rather than silently training on CPU.

To recreate it on a Linux system with a working NVIDIA driver:

```bash
bash setup_cuda_env.sh
```

This creates `.venv`, installs the pinned CUDA training dependencies from `requirements-cuda.txt`, registers the `Python (ATDL Task 1 CUDA)` kernel, and verifies GPU visibility. `requirements-lock.txt` captures the fully resolved environment used for this run.
The completed Task 1 run produced:
`pyproject.toml` and `uv.lock` provide the forward-looking reproducibility manifest for the staged project. Do not run `uv sync` over the completed Task 1 environment unless deliberately recreating it; use `uv lock --check` to verify the lock without modifying the environment.

Run the notebook with that kernel:

```bash
.venv/bin/jupyter lab
```

## Protocol and outputs

The notebook uses a fixed class-balanced 45,000/5,000 train/validation split from the official training data. Checkpoints are selected on validation accuracy; CIFAR-10's official 10,000-image test set is evaluated after selection only.

The completed run produced:

- `resnet34_cifar10_fp32_best.pth` — epoch 190, best validation accuracy 95.54%.
- `resnet34_cifar10_fp32_final.pth` — frozen teacher checkpoint with final test accuracy 95.19% and model metadata.
- `training_history.json` — 200 epochs of train/validation loss, accuracy, generalization gap, and learning rate.
- `Task1_ResNet34_CIFAR10_Teacher.ipynb` — executed notebook including plots and final evaluation.

The final checkpoint has a 10-class CIFAR ResNet34 model state, the training configuration, training-only normalization, split metadata, and explicit teacher provenance. For future tasks, load `model_state_dict`, instantiate the same `ResNet34(num_classes=10)`, call `eval()`, and freeze every parameter.

## Project layout and task gate

## Current completed state

- Task 2 FP32 ResNet18: 95.34% +/- 0.14% validation.
- Task 3 strict ternary ResNet18: 95.21% +/- 0.08% validation.
- Task 4 vanilla ternary KD: 95.12% +/- 0.15% validation.
- DKD: 94.88% three-seed mean, rejected as final candidate.
- DIST: 94.94% one-seed screen, screening-only.
- Final report: `report/final/final_validation_report.pdf`.
- Forensic audit: `final_correctness_audit.md`.

The directories `src/`, `configs/`, `results/`, `plots/`, `hpo/`, `autoresearch/`, and `report/` contain the staged implementations and preserved artifacts. `KD.md` remains the authoritative research roadmap; future extensions must preserve the completed Task 1-4 evidence.
