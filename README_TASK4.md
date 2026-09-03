# Task 4 — Vanilla KD + Ternary QAT (B3), and the pivotal comparison

Drop-in bundle implementing **Task 4 only**: temperature-scaled soft-target
knowledge distillation from the frozen ResNet34 teacher into the ternary ResNet18,
on top of the Task 3 QAT pipeline —

    L = (1 − λ) · CE(z_s, y)  +  λ · T² · KL( softmax(z_t/T) ‖ softmax(z_s/T) )

plus the **mandatory T×λ ablation** and the **T3 (no-KD) vs T4 (KD)** comparison.

Unzip at your repo root. It **replaces** `scripts/train_student_ternary.py` with a
superset (still runs Task 3 exactly, via `--kd-mode none`) and adds the KD module,
the ablation runner, and the comparison script. Depends on Task 2/3 files.

## New / changed files
```
src/kd/vanilla.py                     # KD loss (T² factor), frozen-teacher loader, KD epoch loop
scripts/train_student_ternary.py      # REPLACES Task 3 version; adds --kd-mode vanilla
scripts/ablation_kd_T_lambda.py       # T×λ grid -> results/kd_ablation_T_lambda.csv + heatmap PNG
scripts/compare_baselines.py          # accuracy-ladder decomposition (teacher/FP18/ternary/KD)
configs/kd/resnet18_ternary_vanillaKD.yaml
```

## KD correctness (rubric: "KD formulation & integration", 6 marks)
- **T² factor kept** — soft-target gradients scale ~1/T²; multiplying by T² keeps them comparable to CE so λ and T don't entangle.
- **KL(teacher ‖ student)** via `F.kl_div(log_softmax(student/T), softmax(teacher/T))`.
- **Teacher frozen**: `.eval()` (BN running stats) + `requires_grad=False`, forward under `no_grad`. `assert_teacher_frozen()` enforces it; validated that **no gradient reaches the teacher** (C6/C7).
- Student stays exactly ternary throughout (verified after KD steps).

## Run

First point `--teacher` at your Task 1 ResNet34 checkpoint (e.g.
`resnet34_cifar10_fp32_final.pth`). Then:

```bash
# 0) quick check (1 epoch, 1 seed)
uv run scripts/train_student_ternary.py --kd-mode vanilla --seeds 42 --epochs 5 --smoke \
    --teacher <path/to/resnet34_teacher.pth>

# 1) MANDATORY ablation: T×λ grid (screening budget) -> CSV + heatmap
uv run scripts/ablation_kd_T_lambda.py --teacher <path/to/resnet34_teacher.pth>
#    (defaults: T∈{1,2,4,8,16} × λ∈{0.1,0.3,0.5,0.7,0.9}, 40 epochs/cell, seed 42.
#     Reduce --temperatures/--lambdas/--epochs if compute is tight.)

# 2) full-train the winning (T*, λ*): 3 seeds, 200 epochs
uv run scripts/train_student_ternary.py --kd-mode vanilla \
    --temperature <T*> --kd-lambda <λ*> --teacher <path/to/resnet34_teacher.pth>

# 3) the pivotal comparison (validation): does KD help or hurt the ternary student?
uv run scripts/compare_baselines.py \
    --fp32-student results/resnet18_fp32_summary.json \
    --ternary-nokd results/resnet18_ternary_noKD_summary.json \
    --ternary-kd   results/resnet18_ternary_vanillaKD_ls01_T<T*>_lam<λ*>_summary.json \
    --teacher-acc 95.54

# 4) (optional, recommended) teacher-LS control: repeat 1–3 with an LS=0 teacher.
#    Produce it by re-running your Task 1 training with label_smoothing=0.0, then:
#    ... --teacher <path/to/resnet34_ls0.pth> --teacher-tag ls00
```

Final TEST numbers (report only) come from the firewalled
`scripts/evaluate.py --final-evaluation` on the best checkpoints.

## What it produces
- `experiments/checkpoints/resnet18_ternary_vanillaKD_<tag>_T<T>_lam<λ>_seed{S}.pth`
- `results/best_models/resnet18_ternary_vanillaKD_<tag>_T<T>_lam<λ>_best.pth`
- `results/…vanillaKD…_summary.json`, histories, curves (incl. CE/KL components in history JSON)
- `results/kd_ablation_T_lambda.csv` + `plots/kd_ablation_T_lambda_heatmap.png` (the graded ablation)
- `results/baseline_ladder.json` (the T2→T3→T4 decomposition + T3-vs-T4 delta)

## The pivotal result (report it honestly)
The **Task 4 − Task 3** delta answers whether vanilla KD helps a *ternary* student.
Recent low-bit evidence (SQAKD, QFD) shows naive KD can **underperform** no-KD at
very low precision, so a negative delta is a legitimate, interesting finding — and
the motivation for the Task 5/6 attainable-target methods — not a bug. `compare_baselines.py`
prints the sign either way. Also compare the LS=0.1 vs LS=0.0 teacher: label smoothing
is known to erase the inter-class structure KD relies on (Müller et al. 2019), so the
lower-accuracy LS=0 teacher may distil *better*.

## Verified before delivery (cloud, synthetic data)
- KD loss: λ=0 → CE, λ=1 → T²·KL, KL matches manual T²·KL, KL(identical logits) ≈ 0 ✓
- Teacher loaded frozen + eval; `assert_teacher_frozen` passes ✓
- KD epoch: student updates, **teacher gradients = 0 (no leakage)**, loss/CE/KL finite ✓
- Student remains exactly ternary after KD steps (sparsity ≈ 42%) ✓
- `--kd-mode none` unchanged (Task 3); `vanilla` works; `dkd`/others deferred to Task 5 ✓
- ablation + compare scripts import and route correctly ✓

## Next task
On **"WORK ON TASK 5"**: advanced KD (DKD, DIST, feature/relation, quantized-target)
each with a hypothesis, added behind new `--kd-mode` values on this same trainer.
