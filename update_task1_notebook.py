"""Apply focused, reproducible Task 1 corrections to the supplied notebook."""

import json
from pathlib import Path


path = Path("Task1_ResNet34_CIFAR10_Teacher.ipynb")
notebook = json.loads(path.read_text())
cells = notebook["cells"]


def source(cell_index: int) -> str:
    return "".join(cells[cell_index]["source"])


def replace(cell_index: int, old: str, new: str) -> None:
    value = source(cell_index)
    if old not in value:
        raise RuntimeError(f"Expected text not found in cell {cell_index}")
    cells[cell_index]["source"] = value.replace(old, new).splitlines(keepends=True)


# Make GPU use deliberate rather than silently falling back to CPU.
replace(1, "We set seeds for Python, NumPy, and PyTorch (CPU + CUDA) so runs are repeatable.",
        "We set seeds for Python, NumPy, and PyTorch (CPU + CUDA) so runs are repeatable. "
        "This training run requires an available CUDA GPU; it never silently falls back to CPU.")
replace(3, "from torch.utils.data import DataLoader", "from torch.utils.data import DataLoader, Subset")
replace(5, '    "STRICT_DETERMINISM": False,   # True => bit-exact but slower; False => fast + seeded\n',
        '    "STRICT_DETERMINISM": False,   # True => bit-exact but slower; False => fast + seeded\n'
        '    "REQUIRE_CUDA": True,           # Task 1 is configured to train on the NVIDIA GPU\n'
        '    "ALLOW_TF32": False,            # Keep convolutions/matmuls in IEEE FP32, not TF32\n')
replace(5, '    "COMPUTE_NORM_STATS": False,   # True => compute mean/std from TRAIN split; False => use standard constants\n',
        '    "COMPUTE_NORM_STATS": False,   # True => compute mean/std from TRAIN split; False => use standard constants\n'
        '    "VALIDATION_FRACTION": 0.10,    # stratified held-out training subset; test set stays untouched during fitting\n')

old_device = '''# Device selection.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cuda_version_str():
'''
new_device = '''# Device selection. Task 1 must not accidentally become a many-hour CPU run.
if CONFIG["REQUIRE_CUDA"] and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA was required but is unavailable. Run `nvidia-smi`, activate the CUDA environment, "
        "and restart the kernel; do not train this notebook on CPU."
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    # The teacher is genuinely FP32: disable Ampere TF32 math as well as mixed precision.
    torch.backends.cuda.matmul.allow_tf32 = CONFIG["ALLOW_TF32"]
    torch.backends.cudnn.allow_tf32 = CONFIG["ALLOW_TF32"]
    if not CONFIG["ALLOW_TF32"]:
        torch.set_float32_matmul_precision("highest")

def cuda_version_str():
'''
replace(6, old_device, new_device)
replace(6, 'print(f"Device        : {device}")\n',
        'print(f"Device        : {device}")\nprint(f"Allow TF32    : {CONFIG[\'ALLOW_TF32\']} (False means IEEE FP32 math)")\n')

# Use a fixed, class-balanced validation partition and never select checkpoints on test data.
old_loaders = source(14)
new_loaders = '''# Build a fixed, class-balanced 45k/5k split from the official training set.
# The validation view has no random augmentation; the official test set remains final-evaluation-only.
train_dataset_base = torchvision.datasets.CIFAR10(
    root=CONFIG["DATA_ROOT"], train=True, download=True, transform=train_transform)
validation_dataset_base = torchvision.datasets.CIFAR10(
    root=CONFIG["DATA_ROOT"], train=True, download=True, transform=test_transform)
test_dataset = torchvision.datasets.CIFAR10(
    root=CONFIG["DATA_ROOT"], train=False, download=True, transform=test_transform)

val_per_class = int(len(train_dataset_base) * CONFIG["VALIDATION_FRACTION"] / len(CIFAR10_CLASSES))
if val_per_class <= 0:
    raise ValueError("VALIDATION_FRACTION is too small to allocate one sample per class.")
split_generator = torch.Generator().manual_seed(CONFIG["SEED"])
targets = np.asarray(train_dataset_base.targets)
train_indices, val_indices = [], []
for class_idx in range(len(CIFAR10_CLASSES)):
    class_indices = np.flatnonzero(targets == class_idx)
    permuted = class_indices[torch.randperm(len(class_indices), generator=split_generator).numpy()]
    val_indices.extend(permuted[:val_per_class].tolist())
    train_indices.extend(permuted[val_per_class:].tolist())

train_dataset = Subset(train_dataset_base, train_indices)
validation_dataset = Subset(validation_dataset_base, val_indices)
assert len(train_dataset) + len(validation_dataset) == 50000
assert set(train_indices).isdisjoint(val_indices)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

_use_pin = torch.cuda.is_available()
_nw = CONFIG["NUM_WORKERS"]
_persistent = _nw > 0
loader_generator = torch.Generator().manual_seed(CONFIG["SEED"])
loader_common = dict(
    num_workers=_nw, pin_memory=_use_pin, persistent_workers=_persistent,
    worker_init_fn=seed_worker, generator=loader_generator,
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                          drop_last=False, **loader_common)
validation_loader = DataLoader(validation_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                               drop_last=False, **loader_common)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                         drop_last=False, **loader_common)

print(f"Batch size           : {CONFIG['BATCH_SIZE']}")
print(f"num_workers          : {_nw}")
print(f"pin_memory           : {_use_pin}")
print(f"Train/validation     : {len(train_dataset):,} / {len(validation_dataset):,} (stratified)")
print(f"Official test set    : {len(test_dataset):,} (final evaluation only)")
print(f"Train/val/test batches: {len(train_loader)} / {len(validation_loader)} / {len(test_loader)}")

_xb, _yb = next(iter(train_loader))
print(f"One train batch: images {tuple(_xb.shape)}, labels {tuple(_yb.shape)}")
assert _xb.shape[1:] == (3, 32, 32)
'''
cells[14]["source"] = new_loaders.splitlines(keepends=True)

replace(28, "We now train for `EPOCHS`, recording `train_loss`, `train_acc`, `test_loss`, `test_acc`, and `lr` each epoch into a `history` dict. We **step the scheduler once per epoch** (after the optimizer steps for that epoch).",
        "We now train for `EPOCHS`, recording train and held-out-validation loss/accuracy plus the learning rate each epoch. We **step the scheduler once per epoch** (after the optimizer steps for that epoch).")
start = source(28).find('**On the "validation vs. test" distinction.**')
end = source(28).find('\n\n**Checkpointing.**', start)
if start == -1 or end == -1:
    raise RuntimeError("Validation methodology paragraph not found")
value = source(28)
value = value[:start] + "**Validation protocol.** A fixed, class-balanced 5,000-image validation partition is carved from the official training split. It is used for per-epoch monitoring and checkpoint selection. The official 10,000-image CIFAR-10 test set is evaluated only after the selected checkpoint is frozen, avoiding repeated-test selection bias." + value[end:]
cells[28]["source"] = value.splitlines(keepends=True)

old_loop = source(29)
new_loop = old_loop.replace(
    'history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],\n           "lr": [], "gap": []}\nbest_acc = 0.0',
    'history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],\n           "lr": [], "gap": []}\nbest_acc = 0.0')
new_loop = new_loop.replace('test_loss, test_acc = evaluate(model, test_loader, criterion, device)',
                            'val_loss, val_acc = evaluate(model, validation_loader, criterion, device)')
new_loop = new_loop.replace('gap = train_acc - test_acc', 'gap = train_acc - val_acc')
new_loop = new_loop.replace('history["test_loss"].append(test_loss)\n    history["test_acc"].append(test_acc)',
                            'history["val_loss"].append(val_loss)\n    history["val_acc"].append(val_acc)')
new_loop = new_loop.replace('improved = test_acc > best_acc', 'improved = val_acc > best_acc')
new_loop = new_loop.replace('f"Test Loss {test_loss:.4f} Acc {100*test_acc:5.2f}% | "',
                            'f"Val Loss {val_loss:.4f} Acc {100*val_acc:5.2f}% | "')
new_loop = new_loop.replace('print(f"Best test accuracy: {100*best_acc:.2f}% at epoch {best_epoch}")',
                            'print(f"Best validation accuracy: {100*best_acc:.2f}% at epoch {best_epoch}")')
new_loop = new_loop.replace('"history": history, "best_acc": best_acc, "best_epoch": best_epoch,',
                            '"history": history, "best_val_acc": best_acc, "best_epoch": best_epoch,')
cells[29]["source"] = new_loop.splitlines(keepends=True)

replace(31, 'history["test_loss"]', 'history["val_loss"]')
replace(31, 'history["test_acc"]', 'history["val_acc"]')
replace(31, 'label="Test loss"', 'label="Validation loss"')
replace(31, 'label="Test accuracy"', 'label="Validation accuracy"')
replace(31, 'Training vs. Test Loss', 'Training vs. Validation Loss')
replace(31, 'Training vs. Test Accuracy', 'Training vs. Validation Accuracy')
replace(31, '"Train acc - Test acc"', '"Train acc - Validation acc"')

replace(33, 'history["test_acc"]', 'history["val_acc"]')
replace(33, 'final_test', 'final_val')
replace(33, 'peak_test', 'peak_val')
replace(33, 'test accuracy', 'validation accuracy')
replace(33, 'train-test gap', 'train-validation gap')
replace(33, 'test below its peak', 'validation below its peak')

old_best = source(35)
new_best = '''# Restore the validation-selected checkpoint, then evaluate once on the untouched test set.
best_model = ResNet34(num_classes=10).to(device)
ckpt = torch.load(CONFIG["CHECKPOINT_PATH"], map_location=device)
best_model.load_state_dict(ckpt["model_state_dict"])

best_val_loss, best_val_acc = evaluate(best_model, validation_loader, criterion, device)
best_test_loss, best_test_acc = evaluate(best_model, test_loader, criterion, device)

print("==== Final Teacher Evaluation ====")
print(f"Validation-selected epoch : {ckpt['epoch']}")
print(f"Validation accuracy       : {100*best_val_acc:.2f}%")
print(f"Test accuracy (once)      : {100*best_test_acc:.2f}%")
print(f"Test loss                 : {best_test_loss:.4f}")
'''
cells[35]["source"] = new_best.splitlines(keepends=True)

old_verify = source(43)
new_verify = old_verify.replace('_final_train_acc = history["train_acc"][-1] if history["train_acc"] else float("nan")\n\n', '')
new_verify = new_verify.replace("Training epochs       : {CONFIG['EPOCHS']} (best at epoch {ckpt['epoch']})", "Training epochs       : {CONFIG['EPOCHS']} (validation-selected epoch {ckpt['epoch']})")
new_verify = new_verify.replace("Best test accuracy   : {100*best_test_acc:.2f}%", "Best validation accuracy: {100*best_val_acc:.2f}%\nprint(f\"Final test accuracy  : {100*best_test_acc:.2f}%\")")
new_verify = new_verify.replace("Final train accuracy : {100*_final_train_acc:.2f}%\nprint(f\"Generalization gap   : {100*(_final_train_acc - best_test_acc):.2f} pp\")", "Final train accuracy : {100*history['train_acc'][-1]:.2f}%\nprint(f\"Final train-val gap  : {100*history['gap'][-1]:.2f} pp\")")
cells[43]["source"] = new_verify.splitlines(keepends=True)

replace(45, '"best_test_accuracy": best_test_acc,', '"best_validation_accuracy": best_val_acc,\n        "final_test_accuracy": best_test_acc,')
replace(45, '"history_path": CONFIG["HISTORY_PATH"],', '"history_path": CONFIG["HISTORY_PATH"],\n        "split": {"train": len(train_dataset), "validation": len(validation_dataset), "test": len(test_dataset)},')

replace(46, "For honest tuning, carve a validation split from the training set (e.g., 45k/5k) and select on that; touch the official test set only once at the end. A quick way to add this: use `torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(SEED))` and build a `val_loader`. Keep test evaluation for the final report.",
        "This notebook already uses a fixed class-balanced 45k/5k training/validation split and selects checkpoints by validation accuracy. Keep the official test evaluation as the final report only; if you conduct a new hyperparameter sweep, do not use its test result to select a configuration.")
replace(49, "- [x] Correct 50k / 10k split verified by assertion", "- [x] Correct 50k / 10k official split verified by assertion\n- [x] Fixed class-balanced 45k / 5k training/validation split; no checkpoint selection on test data\n- [x] CUDA is required; the notebook fails rather than silently falling back to CPU\n- [x] IEEE FP32 math enforced (TF32 disabled)")
replace(49, "- [x] Checkpointing on best test accuracy", "- [x] Checkpointing on best validation accuracy; final test set kept untouched during training")
replace(49, "- [x] Test metrics recorded", "- [x] Validation metrics recorded; final test metrics evaluated after model selection")
replace(50, "It records train/test metrics, plots diagnostics, analyzes per-class performance and confusions, restores the best checkpoint, and saves it as the **frozen FP32 teacher**", "It records train/validation metrics, selects the checkpoint on a held-out class-balanced validation split, evaluates the official test set after selection, analyzes per-class performance and confusions, and saves the resulting **frozen FP32 teacher**")

# The dedicated environment kernel is selected automatically in Jupyter.
notebook.setdefault("metadata", {}).setdefault("kernelspec", {}).update({
    "display_name": "Python (ATDL Task 1 CUDA)",
    "language": "python",
    "name": "atdl-task1-cuda",
})

path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Updated {path}")
