"""Repair the two focused edits that require whole notebook code cells."""

import json
from pathlib import Path


path = Path("Task1_ResNet34_CIFAR10_Teacher.ipynb")
notebook = json.loads(path.read_text())
cells = notebook["cells"]


def replace_cell(index: int, value: str) -> None:
    cells[index]["source"] = value.splitlines(keepends=True)


training_cell = "".join(cells[29]["source"])
training_cell = training_cell.replace("best_acc = test_acc", "best_acc = val_acc")
replace_cell(29, training_cell)

replace_cell(43, '''print("==================== FINAL TEACHER VERIFICATION ====================")
print("Architecture         : ResNet34 (CIFAR-adapted stem, BasicBlock [3,4,6,3])")
print(f"Parameters           : {total_params:,} (trainable: {trainable_params:,})")
print(f"Precision            : FP32 "
      f"({'OK' if all(p.dtype == torch.float32 for p in best_model.parameters()) else 'FAIL'})")
print(f"Training epochs      : {CONFIG['EPOCHS']} (validation-selected epoch {ckpt['epoch']})")
print(f"Optimizer            : SGD(momentum={CONFIG['MOMENTUM']}, nesterov={CONFIG['NESTEROV']})")
print(f"Weight decay         : {CONFIG['WEIGHT_DECAY']}")
print(f"Scheduler            : linear warmup ({CONFIG['WARMUP_EPOCHS']}) + cosine decay")
print(f"Base/effective LR    : {CONFIG['LEARNING_RATE']} / {CONFIG['EFFECTIVE_LR']:.5f}")
print(f"Batch size           : {CONFIG['BATCH_SIZE']}")
print(f"Label smoothing      : {CONFIG['LABEL_SMOOTHING']}")
print("Augmentation         : RandomCrop(32, pad=4) + RandomHorizontalFlip")
print(f"Normalization        : mean={tuple(round(m, 4) for m in MEAN)}, std={tuple(round(s, 4) for s in STD)}")
print(f"Best validation acc. : {100 * best_val_acc:.2f}%")
print(f"Final test accuracy  : {100 * best_test_acc:.2f}%")
print(f"Final test loss      : {best_test_loss:.4f}")
print(f"Final train-val gap  : {100 * history['gap'][-1]:.2f} pp")
print("Pretrained weights   : NONE (model built via ResNet34() and randomly initialized)")
print("====================================================================")
print("This model was trained entirely from scratch on CIFAR-10.")
''')

cell28 = "".join(cells[28]["source"])
cell28 = cell28.replace("Whenever test accuracy improves", "Whenever validation accuracy improves")
cells[28]["source"] = cell28.splitlines(keepends=True)

cell50 = "".join(cells[50]["source"])
cell50 = cell50.replace("high test accuracy with stable convergence and a reasonable train/test gap",
                        "high final-test accuracy with stable convergence and a reasonable train/validation gap")
cells[50]["source"] = cell50.splitlines(keepends=True)

cell45 = "".join(cells[45]["source"])
cell45 = cell45.replace('epoch=ckpt["epoch"], best_acc=best_test_acc, config=CONFIG,',
                        'epoch=ckpt["epoch"], best_acc=best_val_acc, config=CONFIG,')
cells[45]["source"] = cell45.splitlines(keepends=True)

path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Repaired {path}")
