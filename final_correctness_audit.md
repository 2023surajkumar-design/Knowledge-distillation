# ATDL Forensic Correctness Audit

Generated: 2026-09-11T06:28:23.969882+00:00

## Executive verdict

**COMPLETE WITH MINOR ISSUES**

Runtime model/data/KD checks pass for the preserved artifacts. Remaining findings are listed below; the official test set remains locked.

Status counts: {'PASS': 32, 'PARTIAL': 0, 'FAIL': 0, 'UNVERIFIED': 0}

## Findings

| Requirement | Status | Severity | Evidence | Required action |
|---|---|---|---|---|
| CIFAR-10 dataset | PASS | none | `src/data.py` | None |
| Frozen split | PASS | none | `src/data.py` | None |
| Test firewall | PASS | none | `src/data.py; scripts/train_student_*.py` | None |
| CIFAR preprocessing | PASS | none | `src/data.py` | None |
| CIFAR ResNet architecture | PASS | none | `src/models/resnet_cifar.py` | None |
| Teacher freeze | PASS | none | `src/kd/teacher.py; scripts/train_student_vanilla_kd.py` | None |
| Vanilla KD formula | PASS | none | `src/kd/losses.py` | None |
| Ternary forward QAT | PASS | none | `src/quant/ternary.py` | None |
| Latent/deployed separation | PASS | none | `src/quant/ternary.py` | None |
| First/final layer coverage | PASS | none | `src/quant/ternary.py; configs/ternary/resnet18_ternary_noKD.yaml` | None |
| STE | PASS | none | `src/quant/ste.py` | None |
| Reproducibility | PASS | minor | `src/training/utils.py; src/data.py` | strict_determinism is false for final runs; report this limitation |
| Duplicate legacy quantizer | PASS | none | `ternary.py; scripts/train_student_*.py` | None |
| Task 2 mean/std | PASS | none | `results/best_models/task2_b1_fp32_resnet18_best.pth` | None |
| Task 3 mean/std | PASS | none | `/home/vu-lab03-pc17/ATDL-1/results/best_models/task3_b2_default_best.pth` | None |
| DKD mean/std | PASS | none | `/home/vu-lab03-pc17/ATDL-1/results/task4/best_models/task5_dkd_final_t2_lam09_a1_b8_r1_best.pth` | None |
| DIST mean/std | PASS | none | `/home/vu-lab03-pc17/ATDL-1/results/task4/best_models/task6_dist_screen_t2_lam09_i1_a1_r1_best.pth` | None |
| Task 4 aggregate | PASS | critical | `results/task4/task4_b3_final_t2_lam09_aggregate_summary.json` | None |
| Ternary checkpoint reload | PASS | none | `experiments/checkpoints/task3_b2_default/resnet18_ternary_seed44.pth` | None |
| Ternary checkpoint reload | PASS | none | `experiments/task4/checkpoints/task4_b3_final_t2_lam09_rerun1/resnet18_ternary_kd_seed42.pth` | None |
| Ternary checkpoint reload | PASS | none | `experiments/task4/checkpoints/task4_b3_final_t2_lam09_remaining_rerun1/resnet18_ternary_kd_seed44.pth` | None |
| Ternary checkpoint reload | PASS | none | `experiments/task4/checkpoints/task5_dkd_final_t2_lam09_a1_b8_r1/resnet18_ternary_kd_seed42.pth` | None |
| Ternary checkpoint reload | PASS | none | `experiments/task4/checkpoints/task6_dist_screen_t2_lam09_i1_a1_r1/resnet18_ternary_kd_seed42.pth` | None |
| Teacher checkpoint reload | PASS | none | `resnet34_cifar10_fp32_best.pth; src/kd/teacher.py` | None |
| Report consistency | PASS | none | `report/final/final_comparison.json; report/final/final_validation_report.md` | None |
| Report rubric coverage | PASS | none | `report/final/final_validation_report.md` | None |
| Notebook clean compilation | PASS | none | `ATDL_post_task4_end_to_end_research.ipynb` | None |
| Research trainer test firewall | PASS | none | `scripts/train_student_fp32.py` | None |
| Research trainer test firewall | PASS | none | `scripts/train_student_ternary.py` | None |
| Research trainer test firewall | PASS | none | `scripts/train_student_vanilla_kd.py` | None |
| Research trainer test firewall | PASS | none | `scripts/run_unified_end_to_end.py` | None |
| Documentation current-state accuracy | PASS | none | `README.md; src/README.md` | None |

## Independently recomputed metrics

- Task 2 raw seeds: [0.9534, 0.9548, 0.952] -> mean 0.953400, sample std 0.001400.
- Task 3 raw seeds: [0.9512, 0.9522, 0.9528] -> mean 0.952067, sample std 0.000808.
- Task 4 raw seeds: [0.9514, 0.9496, 0.9526] -> mean 0.951200, sample std 0.001510; immutable aggregate now records all three.
- DKD raw seeds: [0.9504, 0.9472, 0.9488] -> mean 0.948800, sample std 0.001600.
- DIST raw screen: [0.9494] -> mean 0.949400.

## Rubric estimate

- Ternary quantizer and architecture: **6/6**
- KD formulation and integration: **6/6**
- Joint KD+QAT stability: **6/7** (strict determinism is disabled for final runs)
- Evaluation and compression: **4/5** (teacher comparison is reference-only; no independent teacher validation rerun in this audit)
- Report, ablation, and discussion: **5/6** (report sections now cover architecture, ablation, citations, compression, and limitations; teacher curve/diagram presentation remains limited)
- **Estimated total: 27/30**

## Submission readiness

**Ready after minor fixes.** No expensive retraining is required by this audit unless the instructor requires a fresh teacher evaluation or a single aggregate checkpoint artifact.
