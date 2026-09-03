# Task 4 experiment log

## Protocol lock

Task 4 is restricted to vanilla logit knowledge distillation with strict ternary QAT.  Selection uses only the frozen 45k/5k train/validation split.  The CIFAR-10 test loader is not imported by the Task 4 trainer or preflight.

The immutable B2-default reference is Task 3: mean validation accuracy 95.207% (three seeds) and mean deployed sparsity 47.143%.

## Completed preflight

- Canonical loss tests passed: CE at `lambda=0`, KD at `lambda=1`, teacher-to-student KL direction, `T^2`, and zero KD for identical logits.
- A real 16-image training batch confirmed teacher eval/no-grad behavior, finite student gradients, and 21/21 Conv/FC ternary coverage.
- A one-epoch end-to-end smoke run at T=4, lambda=0.5 completed and saved only validation evidence (`90.52%` initialization sanity check; it is not a comparable final result).

## Next gated experiment

Stage-1 multifidelity grid: T in `{1, 2, 4, 8, 16}` and lambda in `{0.1, 0.3, 0.5, 0.7, 0.9}`, six epochs, seed 42, fixed B2 QAT recipe and Task 2 warm start.  Rank by validation accuracy only; retain top eight for the next stage.  A separate lambda=0 integrity control is retained outside the KD grid.
