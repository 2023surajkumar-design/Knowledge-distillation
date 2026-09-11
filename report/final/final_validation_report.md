# ATDL Final Validation Report

Generated: 2026-09-11T11:44:16.320552+05:30

## Scope and firewall

This report uses the fixed CIFAR-10 45k/5k validation split. The official test set was never used; all source summaries report `test_evaluation: not_run`.

## Final comparison

| Model | Validation mean | Std | Best | Status |
|---|---:|---:|---:|---|
| FP32 ResNet-34 teacher | 95.54% | 0.00% | 95.54% | confirmed reference |
| FP32 ResNet-18 | 95.34% | 0.14% | 95.48% | confirmed |
| Ternary ResNet-18 | 95.21% | 0.08% | 95.28% | confirmed |
| Vanilla ternary KD | 95.12% | 0.15% | 95.26% | confirmed control |
| DKD | 94.88% | 0.16% | 95.04% | rejected |
| DIST | 94.94% | 0.00% | 94.94% | screening |

## Conclusions

- The completed vanilla Task 4 control is the strongest verified ternary KD candidate at 95.12% validation mean.
- DKD is rejected as the final candidate: its three-seed mean is 94.88%, below Task 4.
- DIST remains a one-seed screening result at 94.94%, below Task 4, and is not promoted.
- The ternary no-KD Task 3 control remains a strong reference at approximately 95.21% mean.
- No new method was trained in the finalization pass because no supported remaining method had evidence sufficient to justify GPU expenditure under the time constraint.

## Quantization and compression

The mainline student uses strict ternary Conv/FC deployment with latent FP32 training parameters and QAT/STE. The Task 3 reference reports approximately 47.14% sparsity and 0.5589 mean quantization error. The ideal ternary storage estimate is approximately 20.2x before scale metadata, packing, and runtime overhead. This is not a measured hardware speedup.

## Diagnostics

Existing per-seed diagnostics include CE/KD contributions, teacher/student confidence, entropy, agreement, gradient norms/cosine, learning-rate curves, checkpoint reload checks, sparsity, and quantization error where logged. Missing metrics are not fabricated. Consolidated plots are in `plots/final/`.

## Reproducibility

Best verified final control checkpoint: `experiments/task4/checkpoints/task4_b3_final_t2_lam09_remaining_rerun1/resnet18_ternary_kd_seed44.pth`.

AutoResearch/AutoML remains paused, not abandoned. Future work should resume with one separately reviewed one-seed experiment targeting quantization-compatible knowledge transfer, preserving the test firewall and append-only artifacts.
