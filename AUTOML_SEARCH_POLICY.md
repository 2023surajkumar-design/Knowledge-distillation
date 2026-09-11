# AutoML and AutoResearch policy

## CURRENT RESEARCH EXECUTION STATE

The active mode is **TIME-CONSTRAINED FINALIZATION**. Task 4 is complete at
95.12% +/- 0.15% validation. DKD is rejected at 94.88% mean and DIST remains
screening-only at 94.94%. Broad AutoResearch/AutoML is **PAUSED, NOT
ABANDONED**; future work must resume with a separately reviewed one-seed
experiment rather than repeating completed runs.

## Roles

- **Research planning:** writes a falsifiable hypothesis, control, observable,
  budget, and stopping rule.
- **AutoML / Optuna:** searches bounded numerical ranges only after a method
  works in a deterministic smoke test.
- **Automation:** validates invariants, stores immutable evidence, detects
  failures, and proposes—not executes—follow-ups outside the approved matrix.

AutoML is a **refinement engine, not a hypothesis generator**:

`literature → hypothesis → controlled mechanism experiment → diagnostics →
promotion → bounded AutoML refinement → multi-seed confirmation`.

It may operate only inside an approved experiment family. It must never
silently vary KD method, STE, quantiser, optimizer, scheduler, and QAT schedule
together.

## Hierarchical protocol

| Level | Purpose | Budget / seed policy | Promotion rule |
|---|---|---|---|
| A | numerical + invariant smoke | 1 epoch / 1 seed | finite loss, frozen teacher, full ternary coverage |
| B | method screen | 10 epochs / 1 seed | exceeds matched control by >1 validation example with no safety regression |
| C | shortlist | 30 epochs / 1 seed | gains persist beyond warm-up |
| D | confirmation | 80 epochs / 2 seeds | validation mean exceeds control beyond observed seed noise |
| E | headline | 200 epochs / 3 seeds | mean, std, quantisation, and stability satisfy constitution |

Use TPE plus Hyperband only for a **declared** numerical search.  Persist every
study in SQLite; seed sampler and trials; set a QAT minimum resource of at
least 10 epochs.  Do not tune method identity, quantiser, and schedule in one
unbounded study.

Each study stores search space, sampler and seed, objective, protected
constraints, trial configuration, result, failure reason, contamination status,
and promotion status. Invalid ternary coverage, a changed frozen split, teacher
modification, architecture drift, or a test-loader import invalidates a trial.

## Objective and Pareto record

Primary objective: validation accuracy.  Required secondary columns: seed
standard deviation, sparsity, mean per-layer quantisation error, ternary
coverage, theoretical compression, checkpoint size, training stability, and
runtime.  A candidate that raises accuracy by sacrificing ternary correctness
is invalid, not Pareto-optimal.

Secondary metrics are tracked through a Pareto view; no arbitrary weighted
score is introduced. A lower-accuracy candidate cannot be declared the winner
solely because its secondary diagnostics look better.

## Stopping rules

Stop a branch after two independent screens fail its pre-registered gate or
when its expected gain is below observed seed noise.  Escalate only when an
observable changes coherently and a specific next intervention can distinguish
competing mechanisms.
