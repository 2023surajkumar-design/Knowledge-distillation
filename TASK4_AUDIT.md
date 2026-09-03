# Task 4 audit — vanilla KD under strict ternary QAT

## Frozen references verified

- Task 2: FP32 ResNet18, 95.34% ± 0.14% validation; its seed-43 best checkpoint
  is the canonical QAT warm-start.
- Task 3: strict ternary B2-default, 95.21% ± 0.08% validation. Independent
  reconstruction verifies all 20 Conv2d layers plus the final FC are ternary,
  with 47.12% sparsity for the selected checkpoint. This result is immutable.
- Task 1 teacher: `resnet34_cifar10_fp32_best.pth` is strict-load compatible
  with the Task 1 ResNet34 (21,282,122 parameters; reported 95.54% validation).
- The frozen `src/data.py` 45k/5k split and explicit test-loader firewall are
  reused without modification. Task 4 production modules do not import it.

## Supplied bundle audit

The supplied vanilla loss has the correct KL direction and T-squared factor, but
the associated trainer is not adopted wholesale: it points to stale checkpoint
paths, silently falls back from a missing warm-start, lacks the hardened Task 3
configuration/reload safeguards, and records insufficient KD diagnostics. The
Task 3 trainer therefore remains unchanged for `kd_mode=none`; Task 4 dispatches
to a separate production implementation.

## Task 4 correctness requirements

- Teacher: strict architecture check, `eval()`, `requires_grad=False`, a
  no-grad forward, and an assertion after every backward that teacher gradients
  are absent.
- Student: starts from the frozen Task 2 FP32 checkpoint, then converts to the
  same all-layer ternary QAT model as Task 3 before any KD loss is computed.
- Loss: `(1-lambda) CE + lambda T^2 KL(p_teacher || p_student)`, evaluated as
  `kl_div(log_softmax(student/T), softmax(teacher/T))`.
- Selection: validation only. No test import or test evaluation is permitted.

## Confounds controlled

- The canonical B3 changes only the addition of vanilla logit KD to B2.
- All T/lambda screens use the same seed, QAT setup, augmentation, optimizer,
  schedule, warm-start, and epoch budget; only T/lambda changes.
- Screening is not the headline result. Any finalist is reproduced over seeds
  42/43/44 for 200 epochs before a T3-versus-T4 claim.

## Planned evidence sequence

1. Numerical KD tests, frozen-teacher test, deployed-ternary test, one-batch
   gradient test, and one-epoch production smoke.
2. Lambda=0 integrity control and canonical T=4/lambda=0.5 smoke.
3. Full T in {1,2,4,8,16} × lambda in {0.1,0.3,0.5,0.7,0.9} grid at 5 epochs,
   seed 42; then select a small, predeclared shortlist for longer validation-only
   runs according to stability and meaningful improvement.
4. Freeze one simple configuration and run 200 epochs × seeds 42/43/44.

No Task 5 loss or test-set access is in scope.
