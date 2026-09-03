# Task 3 experiment log

## T3-AUDIT-001 — implementation and protocol audit

- Status: completed before any epoch-scale QAT run.
- Inputs inspected: `KD.md`, frozen Task 2 data/model/training interfaces, Task 2
  checkpoints and summary, every supplied Task 3 file, and the root/bundle
  `ternary.py` comparison.
- Correct: latent/deployed conceptual separation, hard forward assignment,
  per-group shape approach, Task 2 split firewall, and all-conv/FC intent.
- Repaired correctness/reproducibility issues: stale warm-start default;
  config file ignored by trainer; broad diagnostic layer-name matching; verifier
  that only counted unique deployed values; no positive-scale parameterization;
  insufficient checkpoint/run metadata.
- Not changed: Task 2 split, data transforms, FP32 baseline, test firewall, and
  the scientific B2-default quantizer definition.

## T3-SMOKE-001 — one-batch QAT invariants

- Status: passed on CUDA.
- Result: canonical Task 2 warm-start loaded; 20 conv + 1 FC are ternary;
  `conv1` latent weight received a finite nonzero STE update; deployed verifier
  passed; parameter-group invariants passed.
- Interpretation: infrastructure validation only.  This is not reported as an
  accuracy experiment.

## T3-SMOKE-002 — one-epoch trainer (initial attempt)

- Status: infrastructure failure retained as `task3_trainer_smoke`.
- Training and validation completed, but checkpoint reconstruction exposed a CPU
  replacement-layer bug after the base model was moved to CUDA. The repair makes
  each replacement inherit the source layer's device and dtype.
- The reported 91.28% one-epoch validation value is explicitly **not** treated as
  a scientific result. A new run ID will be used for the corrected smoke.

## T3-SMOKE-003 — one-epoch trainer (corrected)

- Status: passed. Seed 42 reached 91.28% validation after one epoch, then its
  saved checkpoint reloaded on CUDA with validation agreement and passed strict
  deployed-weight verification.
- This remains an infrastructure test only; it does not select a quantizer.

## Planned controlled screens

See [experiment matrix](experiments/task3/EXPERIMENT_MATRIX.md).  One seed and
10 epochs is only a screening budget; neither small improvements nor screens
replace the immutable B2-default full 3-seed result.

## T3-S01 through T3-S06 — matched 10-epoch screens

| Condition | Best validation (seed 42) | Decision |
|---|---:|---|
| B2-default | 94.66% | selected for immutable 3-seed B2 |
| B2-random-init | 73.48% | reject; warm-start is necessary at this budget |
| B2-identity-STE | 94.48% | reject; no improvement over clipped STE |
| B2-per-tensor | 94.64% | reject; tied within one example and no advantage |
| B2-learned-symmetric-scale | 93.78% | reject; worse and nearly dense deployment |
| B2-TTQ | 94.12% | reject; worse and nearly dense deployment |

All six checkpoints reloaded with zero validation-accuracy delta and passed the
strict deployed ternary verifier. Full details are in
`results/task3_screening_summary.json`; no test access occurred.
