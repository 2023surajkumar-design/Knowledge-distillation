# KD.md — Evidence-Gated Research Roadmap

## CURRENT RESEARCH EXECUTION STATE

**Mode:** TIME-CONSTRAINED UNIFIED NOTEBOOK EXECUTION.  **Task 4 is in
progress: 1/3 seeds are complete (seed 42); seeds 43 and 44 remain.**  The
completed seed-42 checkpoint, history, configuration, diagnostics and plots
are immutable historical evidence and must not be overwritten or retrained.

Deep AutoResearch/AutoML is **PAUSED, NOT ABANDONED**.  Its hypotheses,
ledger, literature map, policy, failure analysis and database remain the
resume point.  The immediate objective is to complete Task 4 through the
single `ATDL_post_task4_end_to_end_research.ipynb` controller, using the
frozen protocol and validation only; later work resumes the evidence-gated
program from the preserved state.

## Purpose

This is the operational roadmap for knowledge distillation under a strict
ternary-weight QAT constraint. It is not permission to run every named method.
Every phase is conditioned on evidence produced by its predecessor.

The governing rules are in [RESEARCH_CONSTITUTION.md](RESEARCH_CONSTITUTION.md),
the current state is in [POST_TASK4_AUDIT.md](POST_TASK4_AUDIT.md), and the
append-only memory is in `research_db/`.

## Non-negotiable controls

- CIFAR-10 and the frozen stratified 45k/5k train/validation split only.
- The test set is sealed. It is not imported by research training code and is
  used once, only after the final configuration is frozen and explicitly
  authorized.
- Teacher: frozen FP32 CIFAR ResNet34. Student: CIFAR ResNet18. Neither
  architecture can change; no external data or larger student is allowed.
- Every student Conv and FC deployed weight is ternary throughout QAT and at
  inference. Training-only projection/connectors never ship with the student.
- Preserve Task 1–4 results and code paths. New work is append-only, has an
  immutable config, and receives a unique run name.

## Established experimental foundation

| Task | Controlled condition | Validation result | Interpretation |
|---|---|---:|---|
| T1 | FP32 ResNet34 teacher | 95.54% best validation | frozen teacher used for all completed KD runs |
| T2 | FP32 ResNet18 | 95.34% ± 0.14% | student capacity reference and warm start |
| T3 | strict ternary QAT, no KD | 95.21% ± 0.08%; 47.14% sparsity | immutable B2 baseline |
| T4 Stage 3 | vanilla KD, T=2, λ=0.9 | 95.21% ± 0.07% | selected provisional configuration |
| T4 Stage 3 | vanilla KD, T=4, λ=0.9 | 95.14% ± 0.06% | close alternate |
| T4 Stage 3 | vanilla KD, T=8, λ=0.5 | 95.13% ± 0.01% | close alternate |

**Current status:** the fixed T4 final reproduction (T=2, λ=0.9; 200 epochs;
seeds 42/43/44) is incomplete. Seed 42 completed, seed 43 has partial
checkpoint evidence, and seed 44 did not start; no final summary exists.
Partial evidence must be preserved and any completion/reproduction must use a
new run name. No Task-5 training starts until a complete final summary, ternary
verification, and audit update exist.

## What Task 4 taught us

The correct vanilla objective is

\[
L=(1-\lambda)\operatorname{CE}(y,z_s)+\lambda T^2
  \operatorname{KL}(\operatorname{softmax}(z_t/T)\Vert
                     \operatorname{softmax}(z_s/T)).
\]

The teacher-to-student KL direction, T² scaling, λ endpoints, teacher freezing,
and all-layer ternary deployment were tested before screening. A staged 5×5
T×λ grid found that early accuracy was not enough: T=16/λ=0.1 led at six epochs
but weakened at 30 epochs, while T=2/λ=0.9 was the 60-epoch two-seed leader.

The Stage-3 T4/T3 means are essentially tied. Therefore the current evidence
does **not** support the statement “vanilla KD improves strict ternary QAT.”
The final matched reproduction will answer that narrower question. The lower
quantisation error and lower sparsity observed in the vanilla candidate are
diagnostic clues, not causal proof.

## Scientific thesis

The ternary student may not benefit from forcing an exact match to every
continuous teacher signal. The working hypothesis is:

> Knowledge that is relational, normalised, filtered, or otherwise attainable
> under the ternary constraint transfers more reliably than raw FP32 targets.

This makes four discriminating questions more valuable than a large
hyperparameter sweep:

1. Does decoupling target and non-target logits help?
2. Do prediction relations help more than exact KL matching?
3. Does normalised spatial attention help without forcing feature equality?
4. Does batch-wise feature geometry help without forcing raw FP32 features?

## Research operating system

Every experiment is registered in `research_db/experiments.jsonl` with parent,
hypothesis, config, seed, budget, metrics, diagnostic evidence,
interpretation, decision, and next action. The research ledger separates
confirmed, promising, failed, and inconclusive results.

### Research hypothesis → experiment → diagnostics → decision

Every candidate cites a primary ID in
[RESEARCH_HYPOTHESES.md](RESEARCH_HYPOTHESES.md) and declares a parent control,
mechanism, changed variables, and fixed variables. Its promotion path is:

`experiment → accuracy + prediction/KD/gradient/representation/quantisation/
stability diagnostics → mechanism assessment → promotion decision`.

The diagnostic gate asks whether accuracy improved and reproduced; whether the
intended KD signal, agreement, representation alignment, quantisation error,
CE/KD gradient alignment, and stability changed; whether that evidence supports
the proposed mechanism; and whether another variable could explain it. Accuracy
alone is necessary but not sufficient for a mechanism claim.

### Multi-fidelity gates

| Gate | Budget | Seeds | Purpose |
|---|---:|---:|---|
| A | 1 epoch | 1 | unit/invariant smoke: finite loss, frozen teacher, full ternary coverage |
| B | 10 epochs | 1 | screen one isolated hypothesis |
| C | 30 epochs | 1 | reject warm-up-only wins |
| D | 80 epochs | 2 | confirm a stable effect |
| E | 200 epochs | 3 | headline conclusion |

Promotion requires a matched control, >1 validation example screen advantage,
no ternary/stability regression, and a predeclared follow-up. A gain below
observed seed noise is inconclusive. Optuna owns bounded numerical search;
automation owns invariant checks, evidence recording, and failure analysis;
the researcher owns hypotheses and interpretation.

### Task-5 staging and decision colours

| Stage | Purpose | Result status |
|---|---|---|
| 0 | compatibility/sanity: invariants, teacher freezing, reload, ternary coverage | invalid or ready |
| 1 | cheap one-seed mechanism screen | screening only, never “better” |
| 2 | short one-seed stability confirmation | reject warm-up-only wins |
| 3 | longer controlled two-seed confirmation | assess seed stability and mechanism |
| 4 | 200 epochs × three seeds | headline evidence |
| 5 | bounded AutoML refinement | only after promotion |

**GREEN** means a reproducible meaningful improvement with no unacceptable
regression. **YELLOW** means within empirical T3/T4 noise, mixed diagnostics,
or insufficient evidence. **RED** means degradation, instability, invalid
ternary deployment, or a failed intended mechanism. A one- or two-example gain
is a screening signal, not a claim of superiority.

### Diagnostics required for major candidates

- validation accuracy, best epoch, seed mean/std and runtime;
- CE, KD, and each weighted contribution;
- teacher/student accuracy, agreement, entropy, confidence, margin, and
  teacher-correct vs teacher-incorrect subsets;
- per-layer deployed sparsity, positive/negative balance, scale/threshold,
  quantisation error, and dead-layer checks;
- CE/KD/feature/relation gradient norm, variance, cosine and conflict rate;
- strict checkpoint reload and ternary coverage verification.

The failure-analysis table in [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md) maps
observed regressions to minimal targeted follow-ups. It never treats a
correlation as a causal mechanism.

## Task 5 — controlled advanced-KD ladder

**Parent control:** the completed final T4 configuration, not a weaker vanilla
setting. Keep the data, teacher, warm start, QAT recipe, optimizer, schedule,
and total budget fixed. Change one knowledge-transfer mechanism at a time.

### T5.0: post-final diagnostic baseline

Before screening a new loss, reproduce a 10-epoch single-seed vanilla control
with full gradient and representation diagnostics. This is not a selection run;
it establishes the contemporaneous baseline for every T5 screen.

### T5.1: DKD — decoupled logit distillation

**Hypothesis:** separating target-class and non-target-class components lets
the ternary student retain transferable non-target relations without the
coupling in vanilla KL.

**Hypothesis ID:** H1.

**Method:** [DKD](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html), logit-only.
Search only a small predeclared grid of temperature and TCKD/NCKD coefficients
around the vanilla control. Compare DKD against matched vanilla, not T3 alone.

**Falsifier:** no persistent advantage at Gate C, or KD/CE gradient conflict
increases without an accuracy benefit. Do not then stack a feature loss.

### T5.2: DIST — relation-preserving logit KD

**Hypothesis:** with a strong FP32 teacher and ternary student, correlation of
predictions is more attainable than exact softened probabilities.

**Hypothesis ID:** H2.

**Method:** [DIST](https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html), preserving inter-/intra-class prediction relations.
It is tested separately from DKD using a fixed batch size and the same parent.

**Falsifier:** no Gate-C benefit, unstable relation gradients, or no change in
the documented agreement/entropy mismatch.

### T5.3: normalised attention transfer — feature probe

**Hypothesis:** spatial energy maps are a lower-dimensional, architecture-safe
signal that can transfer representation information without requiring a ternary
student to equal FP32 feature amplitudes.

**Hypothesis ID:** H3.

**Method:** one late matched ResNet stage using normalised attention maps from
[attention transfer](https://openreview.net/pdf?id=Sks9_ajex). There is no
projection and no inference-time change.

**Explicit rejection:** raw unnormalised FitNet/MSE is not an initial method;
its scale and dimension mismatch would confound this question.

### T5.4: relational feature KD

**Hypothesis:** batch geometry is more attainable than pointwise feature maps.

**Hypothesis ID:** H4.

**Method:** one late-stage cosine relation matrix or distance/angle loss,
inspired by [RKD](https://arxiv.org/abs/1904.05068), with batch size 128.
Compare feature relation only against the same vanilla parent; do not combine it
with attention transfer in the first ladder.

### T5 screening matrix

| ID | Single changed factor | Gate B | Gate C | Promotion evidence |
|---|---|---:|---:|---|
| V | vanilla diagnostic control | 10e × 1 | — | reference only |
| D | DKD | 10e × 1 | 30e × 1 | accuracy + logit/gradient mechanism |
| R | DIST | 10e × 1 | 30e × 1 | accuracy + relation mechanism |
| A | attention transfer | 10e × 1 | 30e × 1 | accuracy + spatial alignment |
| G | relational feature KD | 10e × 1 | 30e × 1 | accuracy + geometry signal |

At most two Gate-C survivors receive Gate D; only one Gate-D survivor reaches a
200-epoch three-seed conclusion. No component combination is permitted in T5.

## Task 6 — student-aware / quantisation-aware transfer

Task 6 begins only with the confirmed T5 finalist. The first deliverable is a
diagnostic table, not a new loss: per-layer quantisation error/sparsity,
sensitivity proxy, teacher confidence/entropy/correctness, disagreement, and
loss-gradient conflicts.

Then test **one** of the following at a time, with both hypothesised directions
when the direction is unclear:

1. confidence-weighted versus confidence-filtered KD;
2. a stage-wise λ warm-up versus fixed λ;
3. quantisation-error-aware weighting (stronger *and* weaker supervision for
   high-error layers); and
4. a student-aware feature target only if the T5 feature diagnostics show a
   mismatch.

[QFD](https://ojs.aaai.org/index.php/AAAI/article/view/26354) motivates the
last idea but is not copied mechanically: it assumes a quantized/binarized
representation teacher, which this ResNet34 pipeline does not have. Dynamic
temperature, sample-wise λ, target clipping/centering, and cross-layer
connectors remain deferred until their specific diagnostic trigger appears.

## Deferred methods are not forbidden

Raw feature MSE/FitNet, contrastive KD/CRD, large feature connectors, dynamic
temperature, sample-wise lambda, progressive QAT, broad quantiser sweeps, and
giant joint HPO are **deferred, not forbidden**. Any becomes eligible when an
observed, recorded failure mode supplies a direct hypothesis and a clean parent
control—not merely because it is popular or can be searched.

## Task 7 — optimisation and STE

The Task-3 clipped-STE QAT baseline is stable. Therefore identity/alternative
STE, gradient scaling/clipping, scale/threshold learning, and progressive QAT
are conditional interventions—not an automatic sweep. Open this branch only
if the selected KD method shows gradient conflict, dead layers, scale drift,
or non-finite/unstable optimisation. When opened, measure gradient norm,
variance, cosine, quantisation error, sparsity, and accuracy together.

## Task 8 — quantiser interaction

The Task-3 short screen already rejected random initialisation and supplied no
evidence to replace per-channel derived-scale TWN/clipped STE. Revisit
symmetric/asymmetric scales, threshold learning, or TTQ only after a confirmed
KD/quantisation interaction exists. A quantiser change must be compared to
the best KD method and its own no-KD control; otherwise it confounds progress.

## Task 9 — constrained joint AutoML and ablations

Only components with independent Gate-D evidence can be combined. Use a
hierarchical study: method → major KD coefficient → quantiser/schedule → fine
tuning. Persist TPE/Hyperband studies in SQLite and retain every trial. The
final candidate receives removals of each proven component, not a combinatorial
“everything on/everything off” sweep.

## Final evaluation and reporting

Freeze the final configuration by validation mean, seed stability, ternary
verification, sparsity/error, compression, and stability—not best seed.
Then, and only then, run one sealed test evaluation. Report the full ladder:
teacher → FP32 student → ternary/no-KD → vanilla KD → advanced KD →
student-aware KD → final system. Distinguish `log2(3)` theoretical ternary
weight storage from actual PyTorch checkpoint size and from latency; no
hardware-speed claim is valid without an appropriate ternary kernel.

## Required artifacts

```
research_db/                 # append-only experiment memory
research/                    # hypotheses, plans, logs, failure analyses
notebooks/task5_advanced_kd_research.ipynb
notebooks/task6_quantization_aware_kd_research.ipynb
notebooks/task7_ste_research.ipynb
notebooks/task8_quantizer_research.ipynb
notebooks/task9_joint_automl_research.ipynb
experiments/task{5,6,7,8,9}/ results/task{5,6,7,8,9}/ plots/task{5,6,7,8,9}/
```

Every notebook calls production code only. It shows baseline, hypothesis,
matrix, execution, diagnostics, interpretation, and next experiment; it never
contains a duplicate trainer.
