# Post–Task 4 audit

## CURRENT RESEARCH EXECUTION STATE

**TIME-CONSTRAINED UNIFIED NOTEBOOK EXECUTION.** Task 4 is **1/3 seeds
complete**: the preserved clean rerun seed 42 reached 95.14% validation at
epoch 158. Seeds 43 and 44 remain and will be run from the same frozen
protocol in a distinct append-only continuation. The older interrupted seed-43
checkpoint remains an audit artifact only; it is not a completed seed result.

Deep AutoResearch/AutoML is **PAUSED, NOT ABANDONED**. No official test data
may be used. The future resume point is the preserved research ledger/database
and the unified notebook's state checkpoint.

**Audit state:** the Task 4 final 200-epoch × three-seed reproduction is
incomplete. Seed 42 completed and produced a history/diagnostic; seed 43 has
only an append-only best checkpoint (epoch 21, 93.58% validation); seed 44 did
not start. There is no final summary. All conclusions below use completed
validation-only artifacts and will be updated only after a separately named,
non-overwriting completion/reproduction is available. No Task 4 test evaluation
has been performed.

## Established foundation

| Control | Result | Status |
|---|---:|---|
| Task 2 FP32 ResNet18 | 95.34% ± 0.14% validation | immutable reference |
| Task 3 strict ternary QAT/no KD | 95.21% ± 0.08%; 47.14% sparsity; mean layer quantisation error 0.559 | immutable B2 control |
| Task 4 Stage 3, vanilla KD T=2, λ=0.9 | 95.21% ± 0.07%; best 95.26% | selected provisional KD candidate |
| Task 4 Stage 3, vanilla KD T=4, λ=0.9 | 95.14% ± 0.06% | close alternate |
| Task 4 Stage 3, vanilla KD T=8, λ=0.5 | 95.13% ± 0.01% | close alternate |
| Task 4 KD-off integrity control, 30 epochs | 94.98% best validation | objective-path check, not budget-matched to T3 |

The final Task 4 config is frozen in
`configs/kd/resnet18_ternary_vanillaKD_selected.yaml`: Task-2 warm start,
Task-3 per-channel symmetric TWN/derived-scale/clipped-STE QAT, the frozen
Task-1 label-smoothed teacher (`resnet34_cifar10_fp32_best.pth`, SHA-256
`ed19ff5fa087…`), temperature 2, lambda 0.9, and seeds 42/43/44.

## What Task 4 establishes

1. **The vanilla-KD implementation is sound.** Numerical checks validated the
   teacher-to-student KL direction, the T² factor, λ=0/1 limits, frozen teacher
   gradients, and 21/21 ternary Conv/FC layers.  The production smoke run and
   each completed run reload and re-verify the checkpoint.
2. **T and λ interact; temperature cannot be tuned in isolation.** The
   six-epoch 5×5 screen had a transient T=16/λ=0.1 leader (92.92%), but its
   30-epoch result fell to 94.88%.  The stable 60-epoch winner was T=2/λ=0.9.
3. **Vanilla KD is not yet a demonstrated accuracy gain.** T3 is 95.21% ±
   0.08%; the two-seed Stage-3 T4 result is 95.21% ± 0.07%.  The difference is
   effectively zero at the displayed precision and is not a controlled
   200-epoch, three-seed comparison.  The active final run is required before
   making a headline claim.
4. **KD changes the quantisation operating point, but causality is unproven.**
   The Stage-3 winner has ~45.87% sparsity and mean quantisation error ~0.534,
   versus T3's 47.14% and ~0.559.  This motivates a quantisation-aware
   diagnostic; it does not justify changing the quantiser yet.

## Mechanism clues and uncertainties

At the winner's best epochs, validation teacher/student agreement is 95.16–
95.76%; teacher accuracy is 95.54% and student accuracy is 95.16–95.26%.
The student is slightly softer (entropy ≈0.572–0.573 vs teacher ≈0.532;
confidence ≈0.886 vs ≈0.897).  Its converged weighted KD term is small
(≈0.0026–0.0027) relative to weighted CE (≈0.0501).  These observations are
consistent with a mild residual target-calibration/attainability mismatch, but
they do not identify its cause.  Stage-screen runs disabled gradient diagnostics,
so CE/KD gradient conflict is presently **unknown**, not inferred.

## Failures, non-results, and redundant work

- **No scientific training failure occurred in Task 4.** One Stage-1 T=4/λ=0.1
  attempt was interrupted after a checkpoint and was repeated append-only; it is
  an infrastructure event, not a negative method result.
- T=16/λ=0.1 is a useful warning against early-only selection; repeating it at
  the same short budget would be redundant.
- Repeating the entire vanilla T×λ grid, changing the teacher architecture, or
  retraining a label-smoothing-zero teacher is excluded by the immutable
  foundation unless separately authorized.
- Raw unnormalised FitNet matching, contrastive KD, progressive QAT, and broad
  quantiser sweeps are not first-line follow-ups: each adds avoidable confounds
  before the vanilla result or its mechanism is resolved.

## Highest-information Task 5 ladder

All candidates use the final T4 recipe as their parent control, with a
training-only module only where stated.

1. **DKD (first):** tests whether independently weighting target vs non-target
   knowledge improves over the coupled vanilla distribution.  It is logit-only,
   cheap, and isolates a single mechanism.
2. **DIST (second):** tests whether correlation/relations among predictions are
   more attainable than exact probability matching for this strong teacher.
3. **Attention transfer (third):** normalised spatial attention maps at matched
   ResNet stages; dimension-free and no deployed projection.  This is the
   feature-transfer probe retained over raw feature MSE.
4. **Stage-wise relational cosine/RKD (fourth):** tests batch geometry without
   forcing pointwise FP feature equality.

The first screen is one seed, 10 epochs, fixed base recipe and one method
coefficient range per method.  Promote only methods that beat the matched
vanilla control by more than one validation example and have no ternary or
stability regression to 30 epochs; use 80 epochs/two seeds, then 200
epochs/three seeds for any claim.  Feature/attention/relational methods are
never combined in this phase.

## Task 6 gate

Only the Task 5 finalist advances.  First collect layer error/sparsity,
teacher confidence, disagreement, and gradient-cosine diagnostics.  Then test
one factor at a time: confidence-weighted **or** confidence-filtered KD,
lambda warm-up, and quantisation-error-aware loss weighting in both directions.
Dynamic temperature, sample-dependent lambda, and student-aware target
transforms require diagnostic evidence before implementation.

## Decision

Preserve the partial Task-4 evidence; do not overwrite it. Complete or reproduce
the final T4 protocol under a new run name, register it, compare it fairly to
T3, and only then start the DKD screen. The Task 5 matrix is intentionally
small: it maximises information about the proposed attainable-target hypothesis
while protecting the validation budget.

## Mechanism-analysis gate (required before promotion)

Every candidate must answer: (1) did accuracy improve; (2) is it reproduced;
(3) did its intended signal change; (4) did teacher/student agreement change;
(5) did representation alignment change; (6) did quantisation error change;
(7) did CE/KD gradient alignment change; (8) did stability change; (9) does
the evidence support the proposed mechanism; and (10) could an undeclared
variable explain it. Accuracy alone cannot establish a mechanism.

### A. Completed prediction and KD observations

At the Stage-3 T=2/lambda=0.9 best checkpoints, teacher confidence is ~0.897
versus student ~0.886, teacher entropy ~0.532 versus student ~0.572, and
teacher/student agreement is 95.16–95.76%. The student is therefore modestly
softer; it is not accurate to infer whether that is beneficial. At the best
epochs CE is ~0.501 and KD ~0.0029, with weighted CE ~0.0501 and weighted KD
~0.0026–0.0027. These are loss contributions, not an “effective lambda” beyond
the configured lambda=0.9; their ratio is driven by loss scale.

### B. Missing gradient evidence — highest priority

Task-4 screens did not log CE/KD gradient geometry. No statement that vanilla
KD failed because of gradient conflict is currently justified. The next matched
diagnostic baseline must sample `||grad_CE||`, `||grad_KD||`, `||grad_total||`,
the KD/CE norm ratio, CE–KD cosine, and alignment/conflict frequency; optional
layer rows pair each gradient measure with quantisation error and sparsity.
Positive cosine indicates local alignment, near-zero indicates weak coupling,
and negative cosine indicates sampled objective conflict—not a causal proof.

### C. Quantisation and representation evidence

The completed T4 Stage-3 winner has ~45.87% sparsity and mean layer error ~0.534
against T3's 47.14% and ~0.559. Layer-wise scale/threshold, latent/deployed
difference, zero-balance, and dead-channel checks are required before a
quantisation-bottleneck explanation. Representation diagnostics must report
normalised feature cosine, feature norms, and class-centroid similarity at a
declared compatible stage. Optional CKA is deferred unless its cost is
justified; feature similarity is never treated as an accuracy surrogate.

The explicit outstanding question is: **did vanilla KD remain neutral because
its signal was weak, because CE/KD conflicted, because the student could not
represent the teacher, or because the ternary bottleneck dominated?**
