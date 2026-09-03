# Task 3 experiment matrix

All rows use the frozen Task 1 split, CIFAR-10 training/validation only, all
convolution and FC weights ternary, and no KD.  No row may call the test loader.

| ID | Parent | Condition | Controlled change | Budget | Hypothesis | Acceptance rule |
|---|---|---|---|---:|---|---|
| T3-SMOKE | none | B2-default | canonical config | one batch + one epoch | implementation is sound | strict verifier, reload, finite loss |
| T3-S01 | B2-default | B2-default | none | 10 ep, seed 42 | establishes learning curve | required reference; not selection evidence |
| T3-S02 | B2-default | B2-random-init | remove only warm-start | 10 ep, seed 42 | warm-start reduces QAT optimization burden | record delta; no baseline replacement |
| T3-S03 | B2-default | B2-identity-STE | change only STE | 10 ep, seed 42 | clipping stabilizes coarse gradients | shortlist only if stable and clearly better |
| T3-S04 | B2-default | B2-per-tensor | change only scale scope | 10 ep, seed 42 | per-channel reduces quantization error | diagnostic comparison; compliant but not default |
| T3-S05 | B2-default | B2-learned-symmetric-scale | derived to learned symmetric scale | 10 ep, seed 42 | trainable scale may recover error | must avoid scale collapse and show material gain |
| T3-S06 | B2-default | B2-TTQ | learned asymmetric scale + TTQ threshold | 10 ep, seed 42 | asymmetry may improve representational fit | ablation only; must retain exact ternary deployment |
| T3-C01 | shortlisted rows | serious candidate | frozen after screening | 40 ep, seeds 42/43 | early result reproduces | consistent validation/stability over two seeds |
| T3-FINAL | none | B2-default | immutable canonical baseline | 200 ep, seeds 42/43/44 | quantify pure ternarization cost | complete only after all preceding evidence is logged |

The canonical B2-default remains per-channel, symmetric, TWN (0.7 mean absolute
weight), derived detached alpha, clipped STE, warm-started from Task 2, and all
conv/FC ternary.  It is never replaced by an ablation result.
