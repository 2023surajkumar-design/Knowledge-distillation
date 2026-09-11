# ATDL Final Validation Report

Generated: 2026-09-11T12:21:06.810305+05:30

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
| DIST T=1 lambda=0.5 continuation | 95.02% | 0.00% | 95.02% | provisional negative |

## Conclusions

- The completed vanilla Task 4 control is the strongest verified ternary KD candidate at 95.12% validation mean.
- DKD is rejected as the final candidate: its three-seed mean is 94.88%, below Task 4.
- DIST remains a one-seed screening result at 94.94%, below Task 4, and is not promoted.
- The bounded DIST T=1/lambda=0.5 continuation reached 95.02% at epoch 48 in one seed, below the Task 4 mean, and remains provisional/negative.
- The ternary no-KD Task 3 control remains a strong reference at approximately 95.21% mean.
- No new method was trained in the finalization pass because no supported remaining method had evidence sufficient to justify GPU expenditure under the time constraint.

## Architecture and method

The teacher is a CIFAR-adapted ResNet-34 with a 3x3 stride-1 stem, no ImageNet max-pool, and a 10-class classifier. The student is the corresponding CIFAR ResNet-18. Every student convolution and final fully-connected weight is converted to per-output-channel symmetric TWN-style weights. For latent weight W, Delta = 0.7 mean(|W|) per output channel, Q(W) is in (-1, 0, 1), and deployed W_hat = alpha Q(W), where alpha is the mean active absolute latent weight. BatchNorm parameters and biases remain FP32; Conv/FC weight matrices are ternary. The forward path uses W_hat throughout QAT, while the optimizer updates latent FP32 parameters through a clipped STE.

The vanilla KD loss is (1-lambda) CE + lambda T^2 KL(teacher || student), with T=2 and lambda=0.9 in the completed Task 4 control. The teacher is eval-mode, frozen, excluded from the optimizer, and evaluated under torch.no_grad().

## KD hyperparameter ablation

The validation-only staged ablation selected T=2, lambda=0.9: 92.90% after 6 epochs, 95.16% after 30 epochs, and 95.21% +/- 0.07% after the 60-epoch Stage-3 screen. Nearby controls were T=4, lambda=0.9 at 95.14% +/- 0.06%, T=8, lambda=0.5 at 95.13% +/- 0.01%, and a lambda=0 integrity control at 94.98% best validation. These are validation-only screening results.

## Required references

He et al. (ResNet, CVPR 2016); Hinton, Vinyals, Dean (KD, NeurIPS Workshop 2015); Li, Zhang, Liu (Ternary Weight Networks, 2016); Li, Zhang, Liu (Trained Ternary Quantization, ICLR 2017); Bengio, Leonard, Courville (STE, 2013); Yin et al. (STE analysis, ICLR 2019).

## Limitations

Final CUDA runs use strict_determinism=false, while seeds, split manifests, configurations, runtime metadata, and reload checks are recorded. Theoretical ternary compression excludes scale metadata, packing, and kernel overhead; no real hardware speedup is claimed. Representation similarity was not logged and is unavailable. The official test set remains locked.

## Quantization and compression

The mainline student uses strict ternary Conv/FC deployment with latent FP32 training parameters and QAT/STE. The Task 3 reference reports approximately 47.14% sparsity and 0.5589 mean quantization error. The ideal ternary storage estimate is approximately 20.2x before scale metadata, packing, and runtime overhead. This is not a measured hardware speedup.

## Diagnostics

Existing per-seed diagnostics include CE/KD contributions, teacher/student confidence, entropy, agreement, gradient norms/cosine, learning-rate curves, checkpoint reload checks, sparsity, and quantization error where logged. Missing metrics are not fabricated. Consolidated plots are in `plots/final/`.

## Reproducibility

Best verified final control checkpoint: `experiments/task4/checkpoints/task4_b3_final_t2_lam09_remaining_rerun1/resnet18_ternary_kd_seed44.pth`.

AutoResearch/AutoML remains paused, not abandoned. Future work should resume with one separately reviewed one-seed experiment targeting quantization-compatible knowledge transfer, preserving the test firewall and append-only artifacts.
