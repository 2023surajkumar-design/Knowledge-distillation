# Task 4 report — in progress

## Scope

This report will describe validation-only vanilla logit KD for strict ternary ResNet18 QAT.  It will not report test metrics until a protocol-compliant, final evaluation is separately authorized.

## Fixed ingredients

- Student: Task 2 ResNet18 FP32 warm start, then all convolution and FC weights converted to strict per-channel symmetric TWN with derived scale and clipped STE.
- Teacher: frozen Task 1 ResNet34 checkpoint `resnet34_cifar10_fp32_best.pth`, selected by validation accuracy (95.54%).
- Canonical candidate: temperature 4, lambda 0.5; CE label smoothing 0.1; SGD/warmup/cosine recipe frozen from Task 3.
- Loss: `(1-lambda) CE + lambda*T^2*KL(softmax(teacher/T) || softmax(student/T))`.

Results will be appended only after the staged validation search and the three-seed final protocol complete.

## Validation-only selection (pre-final)

The staged `T × lambda` screening selected `T=2`, `lambda=0.9`.

| Stage | Candidate | Validation result |
|---|---|---:|
| Stage 1 (6 epochs, seed 42) | T=2, lambda=0.9 | 92.90% |
| Stage 2 (30 epochs, seed 42) | T=2, lambda=0.9 | 95.16% |
| Stage 3 (60 epochs, seeds 42/43) | T=2, lambda=0.9 | 95.21% ± 0.07% |

The closest Stage-3 alternatives were T=4/lambda=0.9 (95.14% ± 0.06%) and
T=8/lambda=0.5 (95.13% ± 0.01%).  A lambda=0, 30-epoch integrity control
completed at 94.98% best validation.  These are all validation-only results.
