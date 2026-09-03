# Phase 0 Audit — Task 1 Baseline and Project Readiness

**Audit date:** 2026-08-28  
**Source of truth:** [`KD.md`](KD.md)  
**Scope:** every tracked source file, the executed Task 1 notebook, both checkpoints, stored history, dataset layout, Git history, and runtime environment. No student training, quantization, KD, HPO, or new test evaluation was run during this audit.

## 1. Executive status

Task 1 is complete. It is a CIFAR-adapted, randomly initialized FP32 ResNet-34 teacher trained on a fixed class-balanced 45,000/5,000 split of CIFAR-10's official training data. The selected epoch is 190/200, with **95.54% validation accuracy** and a previously recorded final **95.19% official-test accuracy**. This is the valid B0 teacher baseline.

The repository is **not yet an end-to-end KD/QAT project**. It contains no ResNet-18 student, ternary forward path, STE, KD loss, student trainer, ternary verifier, compression analysis, HPO study, or student results. Accordingly, B1–B4 and Tasks 2–12 remain incomplete.

The project plan explicitly says to execute one task at a time. The immediate next research action is **Task 2**, not Task 3+.

## 2. Repository inventory

| Path | Status / purpose |
|---|---|
| `KD.md` | Complete 407-line master plan; currently untracked and must be retained/versioned. |
| `README.md` | Accurate Task 1-only run instructions and baseline results; needs expansion only once later tasks exist. |
| `Task1_ResNet34_CIFAR10_Teacher.ipynb` | Executed Task 1 implementation and embedded figures/output. |
| `training_history.json` | 200-epoch train/validation metrics, LR, config, elapsed time. |
| `resnet34_cifar10_fp32_best.pth` | Validation-selected epoch-190 checkpoint, ignored by Git. |
| `resnet34_cifar10_fp32_final.pth` | Frozen teacher inference checkpoint, ignored by Git. |
| `requirements-cuda.txt` / `requirements-lock.txt` | CUDA environment intent plus a resolved Task 1 environment. |
| `setup_cuda_env.sh` | Linux CUDA virtual-environment/bootstrap script. |
| `update_task1_notebook.py`, `repair_task1_notebook.py` | Historical one-shot notebook migration helpers; not runtime training code. |
| `finalize_teacher_metadata.py` | Repairs the final checkpoint selection metadata; made explicit about `weights_only=False`. |
| `data/` | CIFAR-10 Python batches plus the source tarball; the raw batches are committed despite the intended future data-gitignore policy. |
| `src/`, `configs/`, `checkpoints/`, `results/`, `plots/`, `hpo/`, `autoresearch/`, `report/` | Task-gated scaffold created in this Phase 0 pass. |

Git has one commit, `19df586 Task 1` (2026-08-28). The working tree contained the untracked `KD.md` before this audit. Teacher checkpoints and the CIFAR archive are intentionally ignored by the current `.gitignore`.

## 3. Reproducible B0 baseline

| Item | Exact baseline |
|---|---|
| Dataset | CIFAR-10 only; 50,000 official training images and 10,000 official test images |
| Split | fixed, class-balanced, seed 42: 45,000 train / 5,000 validation / 10,000 official test |
| Input / classes | 3×32×32, 10 classes |
| Normalization | mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)` |
| Training augmentation | `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` |
| Validation/test transform | tensor conversion and normalization only |
| Teacher | CIFAR ResNet-34, BasicBlock depths `[3,4,6,3]`, 3×3 stride-1 stem, no max-pool, global average pool, 10-way FC |
| Parameters | 21,282,122 trainable FP32 parameters |
| Compute | 1.1594 GMAC/image, or 2.3188 GFLOP/image when multiply+add counts as two FLOPs |
| Optimizer | SGD, momentum 0.9, Nesterov enabled, weight decay `5e-4` |
| LR schedule | base/effective LR 0.1; linear warmup for 5 epochs then cosine decay over 200 epochs |
| Loss | cross entropy with label smoothing 0.1 |
| Batch size / workers | 128 / 4 |
| Initialisation | Kaiming-normal Conv, BN gamma=1/beta=0, Linear normal std=0.01/bias=0 |
| BatchNorm | normal train/eval behavior; no frozen BN, EMA, dropout, mixed precision, gradient clipping, Cutout, Mixup, CutMix, or RandAugment |
| Selection protocol | highest validation accuracy; test evaluation after checkpoint selection |
| Best epoch / validation | epoch 190; 95.54% |
| Recorded final test | 95.19%; loss is stored in the final checkpoint |
| Last-epoch train/val | 99.99% / 95.40%; gap 4.59 percentage points |
| Hardware at audit | NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB VRAM, compute capability 8.9 |
| Runtime at audit | Ubuntu 24.04.4, Python 3.13.9, torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8 |

`training_history.json` agrees with the documented run: 200 entries per metric, maximum validation accuracy 0.9554 at epoch 190, final validation accuracy 0.9540, and LR decayed to `6.48875e-06`.

## 4. Execution and dependency trace

```text
raw CIFAR-10
  -> size/shape assertions and visual sample grid
  -> fixed CIFAR normalization
  -> train augmentation / deterministic class-balanced index split
  -> 45k augmented train loader + 5k clean validation loader + 10k clean test loader
  -> randomly initialized CIFAR ResNet-34
  -> smoothed CE + SGD/Nesterov + warmup/cosine
  -> epoch train + validation metrics
  -> best-validation checkpoint at epoch 190
  -> restore selected checkpoint
  -> final test metrics, per-class metrics, confusion matrix, example predictions
  -> final teacher checkpoint + history JSON
```

The intended downstream branches—FP32 ResNet-18, ternary ResNet-18 QAT, frozen-teacher KD, advanced KD, and compression analysis—do not exist yet. This is correct for the Task 1-only repository, but it is the main gap relative to the complete project pipeline.

## 5. Notebook cell audit

All 51 cells were inspected. Markdown cells explain the adjacent code; the table records every executable cell and whether its result is consumed.

| Cell | Function, inputs, outputs, and audit conclusion |
|---:|---|
| 3 | Imports runtime, PyTorch, torchvision, plotting, optional tqdm/sklearn. Required by all later cells; fallback behavior is sensible. |
| 5 | Defines all Task 1 hyperparameters, derived effective LR, worker policy, and class names. Consumed throughout. Correct; the `STRICT_DETERMINISM=False` setting is seeded but not bit-exact. |
| 6 | Seeds Python/NumPy/PyTorch, enforces CUDA, disables TF32, and prints runtime details. Correct. Fast cuDNN mode intentionally trades exact repeatability for speed. |
| 8 | Downloads/loads raw CIFAR-10 and asserts 50k/10k, 10 classes, and 3×32×32 tensors. Correct integrity gate; raw datasets are later re-opened with training transforms. |
| 10 | Displays randomly chosen raw images. It consumes Python RNG but has no model-selection or data-split effect because the split uses a separately seeded Torch generator. Exploratory only. |
| 12 | Defines standard CIFAR statistics; optional training-split mean/std implementation; defines train and eval transforms. Correct. The baseline uses public full-training-set constants, consistently across splits. |
| 14 | Creates two transformed views of training data; makes an exactly balanced 4,500/500-per-class split using seed 42; constructs loaders. Correct and no test leakage. The split indices are reproducible from code but not persisted explicitly—future `data.py` should persist/fingerprint them. |
| 16 | Implements BasicBlock and CIFAR ResNet-34. Correct original post-activation BasicBlock structure: Conv-BN-ReLU, Conv-BN, identity/projection-BN, add-ReLU. Downsampling and channel changes are correct. |
| 18 | Instantiates the model; reports parameter count and checks FP32 and `[B,10]` output. Correct. This output is used later for final verification/metadata. |
| 20 | Creates label-smoothed CE. Correct for teacher training; the LS=0.1 choice is a planned teacher-quality control in Task 4. |
| 22 | Defines SGD/Nesterov and LambdaLR warmup/cosine. Correct; scheduler is stepped once per epoch after optimizer updates. |
| 23 | Plots the mathematical LR schedule without mutating training state. Useful diagnostic; no redundant training work. |
| 25 | Per-epoch train loop with correct `train()`, zero-grad, forward/backward/update, sample-weighted loss, and accuracy. No AMP/gradient clipping. |
| 27 | Evaluation loop uses `@torch.no_grad()` and `eval()`, returns optional predictions. Correct and numerically stable for this scale. |
| 29 | Main 200-epoch train/validation loop; saves the best validation checkpoint, history, and elapsed time. Correct selection logic. The checkpoint schema must be kept stable for later loaders. |
| 31 | Generates loss, accuracy, generalization-gap, and LR figures. Outputs were embedded in the notebook but not exported to `plots/`. |
| 33 | Interprets train/validation gap. The diagnostic wording is reasonable; it is not used to alter the completed run. |
| 35 | Restores the validation-selected checkpoint and computes final validation/test metrics. Correct. Explicit `weights_only=False` should be used in new code under PyTorch 2.6+. |
| 37 | Computes per-class accuracy on the frozen final teacher. Correct reporting, not used for selection. |
| 39 | Computes sklearn/manual confusion matrix and normalized display. Correct reporting only. |
| 41 | Displays a fixed unshuffled test batch with predictions. Correct visualization only. |
| 43 | Prints concise final provenance/quality summary. Correct except it depends on the stored history in the notebook session. |
| 45 | Saves/reloads the final teacher and verifies equal test accuracy. Correct intent, but the materialized checkpoint initially had missing top-level selection metadata; repaired below. |
| 48 | Defines optional Mixup helpers. They are not called, so they do not affect B0 and should not be described as part of the baseline. |

## 6. Teacher architecture and quality assessment

The implementation is a valid CIFAR adaptation of ResNet-34, not an accidental ImageNet stem: its 3×3 stride-1 stem and omission of max-pool preserve 32×32 spatial detail. It has 16 BasicBlocks (32 convolutional layers inside blocks), plus the stem and FC, matching the conventional ResNet-34 layer count. Projection shortcuts are used only when stride or channels differ. Global average pooling produces 512 features for the 10-class FC. Initialization is appropriate.

The stored learning curves indicate a well-optimized teacher rather than obvious undertraining: near-saturated training accuracy, 95.54% peak validation, and a 4.59pp final train–validation gap. That gap indicates modest late-stage overfitting, but checkpoint selection at epoch 190 avoids choosing the final epoch by default. There is no evidence of hidden test-driven model selection.

Quality measures available today are accuracy/loss/gap, per-class accuracy, confusion matrix, and visual predictions in the notebook. Calibration/ECE, confidence entropy, top-1/top-2 margins, augmentation robustness, and explicit FLOP measurement were not saved. They must not be retroactively computed on the official test set now: future final evaluation should collect all final-report metrics in one fixed pass after selection. Teacher calibration is especially relevant because this teacher used label smoothing 0.1; `KD.md` correctly schedules an LS=0 teacher control for Task 4 use, not for premature implementation now.

## 7. Checkpoint provenance and repair

| Artifact | Baseline identity |
|---|---|
| Final checkpoint | `resnet34_cifar10_fp32_final.pth` |
| Role | frozen FP32 ResNet-34 teacher |
| Epoch / selection | 190 / validation accuracy |
| Validation / test | 0.9554 / recorded 0.9519 |
| Parameters | 21,282,122 |
| Final SHA-256 after metadata repair | `75dbc60f1a287437ca37e231e4ab9e02dc09402c7a4c6d43cc858ca0428c7b42` |

**Repair made:** the saved final checkpoint had `best_accuracy: null` and no `selection_metric` despite valid explicit accuracy fields. Only metadata was changed: `best_accuracy=0.9554` and `selection_metric="validation_accuracy"`. Tensor weights, optimizer state, scheduler state, epoch, and metrics were not altered. The repair script now calls `torch.load(..., weights_only=False)` explicitly.

## 8. Problems, risks, and required controls

1. **Strict test-pass discipline:** the notebook calls the test loader more than once for final reporting (metrics, per-class data, and reload check). No decision uses those outputs, so this is not selection leakage; nevertheless, future tasks must compute all final metrics in one frozen final-evaluation routine and never run test during tuning.
2. **Metadata issue repaired:** see §7. New checkpoints must include a selection metric and non-null best validation score by schema validation.
3. **Historical mutator scripts:** `update_task1_notebook.py` and `repair_task1_notebook.py` can rewrite the notebook and should never be part of an experiment run. Retain them as provenance or archive them later, but do not execute them casually.
4. **Dataset source control:** CIFAR batch files are already tracked. Do not rewrite history or delete them in this audit; future fetched/generated data should remain ignored.
5. **Reproducibility boundary:** the seed is fixed but `STRICT_DETERMINISM=False`; 3-seed conclusions are required for later headline results. Future shared split code must be frozen and its indices/fingerprint persisted.
6. **Environment:** the old requirements workflow is valid for this machine. A `pyproject.toml` and `uv.lock` were added for a reproducible future workflow; no package was added to the Task 1 venv beyond the requested `uv` CLI installed to the user tool location.
7. **Teacher suitability:** ResNet-34 is a strong, valid teacher for the fixed ResNet-18 student. Do not change teacher architecture. The only planned teacher control is the same architecture with label smoothing 0.0, introduced in the prescribed Task 4 comparison.

## 9. Completed / incomplete task ledger

| Status | Work |
|---|---|
| Complete | Task 1 / B0: FP32 CIFAR ResNet-34 teacher and reproducible validation-selected checkpoint |
| Prepared only | Task 2 folder/interfaces; no code or run exists |
| Incomplete | B1 FP32 ResNet-18 no-KD, B2 ternary QAT no-KD, B3 vanilla KD+QAT, B4 advanced KD+QAT |
| Incomplete | ternary verifier, compression analysis, Optuna study, autoresearch harness, student checkpoints, report assets |

## 10. Immediate next step

On an explicit **“WORK ON TASK 2”** instruction: implement and freeze the shared CIFAR-10 split/data module and CIFAR ResNet definitions, then train only the FP32 ResNet-18 no-KD baseline under the B0 recipe. Preserve the listed teacher checksum, compare on validation only, and defer all ternarization/KD/test reporting until their task gates.
