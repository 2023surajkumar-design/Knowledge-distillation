# Task 2 — FP32 ResNet18 Baseline (B1)

Integrated implementation of **Task 2 only** from `KD.md`: a full-precision, from-scratch
ResNet18 CIFAR-10 baseline with **no KD** and **no quantization**, sharing the frozen data
split and the exact training recipe of the Task 1 teacher so the teacher↔student capacity
gap is a controlled comparison.

## Implemented layout

```
atdl-kd-ternary/
├── src/
│   ├── data.py                     # FROZEN shared CIFAR-10 pipeline + 45k/5k split + test firewall
│   ├── models/resnet_cifar.py      # shared ResNet18 + ResNet34 (Task-1-checkpoint compatible)
│   └── training/utils.py           # seeding, warmup+cosine, train/eval, checkpoint, plots
├── scripts/
│   ├── train_student_fp32.py       # TASK 2 trainer (3 seeds)
│   ├── evaluate.py                 # FINAL test eval (firewalled; --final-evaluation required)
│   └── run_smoke_test.py           # 1-epoch pipeline check
└── configs/student/resnet18_fp32.yaml
```

`src/data.py` and `src/models/resnet_cifar.py` are the shared interfaces for Tasks 3–12. They were
verified against the Task 1 split and teacher checkpoint before integration, and must be treated as
frozen once B1 is complete. The original external bundle remains in `atdl_task2_bundle/` as input
provenance; only the root `src/` and `scripts/` paths are executable project code.

## Run

```bash
# 0) infrastructure smoke: real CIFAR subset, no test access and no scientific result
uv run --locked scripts/run_smoke_test.py

# 1) Task 2 baseline: 3 seeds (42, 43, 44), 200 epochs each (~30–40 min/seed on a 4080)
uv run scripts/train_student_fp32.py

#    (mandatory short real-CIFAR preflight before B1; output is immutable and separate)
uv run --locked scripts/train_student_fp32.py --seeds 42 --epochs 5 --smoke

# 2) FINAL test numbers — ONLY when writing the report (unlocks the test set)
uv run --locked scripts/evaluate.py \
    --checkpoint experiments/checkpoints/task2_b1_fp32_resnet18/resnet18_fp32_seed42.pth \
                 experiments/checkpoints/task2_b1_fp32_resnet18/resnet18_fp32_seed43.pth \
                 experiments/checkpoints/task2_b1_fp32_resnet18/resnet18_fp32_seed44.pth \
    --arch resnet18 --final-evaluation
```

## What it produces
- `experiments/checkpoints/<run-name>/resnet18_fp32_seed{42,43,44}.pth` — per-seed best-on-validation
- `results/best_models/<run-name>_best.pth` — convenience copy of the validation-best seed
- `experiments/histories/<run-name>/resnet18_fp32_seed{S}.json` — per-epoch metrics and runtime metadata
- `plots/<run-name>/resnet18_fp32_seed{S}_{loss,acc,gap,lr}.png` — curves
- `results/task2_final_summary.json` — B1 three-seed validation summary
- `results/task2_test_evaluation.json` — final independent checkpoint test results only after step 2

The trainer refuses to overwrite any checkpoint, history, best-model copy, or summary. Use a new
`--run-name` for a diagnostic rerun. `--smoke` requires an explicit shorter `--epochs` value; it
uses the real CIFAR pipeline and merely labels the output as a preflight rather than B1.

## Constraints honored
- **C1/C8** CIFAR-10 only, no external data.
- **C9/C11** test set locked behind `get_test_loader(final_evaluation=True)`, reachable only via `evaluate.py --final-evaluation`. Training selects on the 5k validation split.
- **C10** identical stratified 45k/5k split to Task 1 (seed 42, 500 val/class), reproduced exactly.
- **Scope** Task 2 only: `--kd` is a placeholder that raises (KD is Task 4); no quantization anywhere.

## Verified before delivery (cloud smoke test on synthetic data)
- Split: 45000/5000, 500 val/class, disjoint, deterministic ✓
- Firewall refuses the test loader without the explicit flag ✓
- ResNet18 = 11,173,962 params · ResNet34 = 21,282,122 (matches T1) ✓
- Train/eval loop, warmup+cosine schedule, checkpoint reload-identical, 4 PNG curves — verified locally before B1
- `--kd` guard raises; `evaluate.py` refuses without `--final-evaluation` ✓
- *(CIFAR download itself couldn't run in the cloud sandbox — proxy-blocked — but it's stock torchvision and runs on your machine; `run_smoke_test.py` confirms it there.)*

## Expected outcome (not a claim)
A well-trained FP32 ResNet18 with crop+flip on CIFAR-10 typically lands ~94–95% test accuracy,
i.e. ~0.5–1.5 pp below your 95.19% ResNet34 teacher — that gap is the **capacity** component,
which Task 3 (ternarization) and Task 4+ (KD) then probe separately. Your exact number depends
on hardware, library versions, and seed.

## Next task
On **"WORK ON TASK 3"**: implement the ternary quantizer (`src/quant/`), the ternary ResNet18,
STE, and the no-KD QAT baseline (B2) — reusing this exact split and evaluation firewall.
