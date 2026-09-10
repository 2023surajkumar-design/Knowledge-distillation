# Research ledger

## CURRENT RESEARCH EXECUTION STATE

**TIME-CONSTRAINED UNIFIED NOTEBOOK EXECUTION:** Task 4 is 1/3 complete
(seed 42); seeds 43 and 44 remain. The frozen, completed seed is preserved as
the baseline anchor. Deep AutoResearch/AutoML is **PAUSED, NOT ABANDONED**;
the documents and `research_db/` are its durable continuation point.

## Confirmed

- The Task-3 B2 baseline is strict all-Conv/FC ternary QAT and reload-verifies.
- FP32 Task-2 warm-start is essential: the Task-3 10-epoch random-init probe
  reached 73.48%, versus 94.66% for the matched warm-start probe.
- Per-channel derived-scale symmetric TWN with clipped STE is the established
  Task-3 quantisation control (95.21% ± 0.08% validation).
- Task-4 vanilla KD preserves strict ternary deployment and teacher isolation.
- T and lambda must be selected jointly; a short-horizon winner did not remain
  the long-horizon winner.

## Promising

- Vanilla KD T=2/lambda=0.9 is the Stage-3 leader (95.21% ± 0.07%), but not
  yet a demonstrated improvement over T3.  The 200-epoch × three-seed final is
  active.
- The KD winner's lower quantisation error and sparsity suggest a measurable
  target/quantiser interaction worth diagnosing, not yet optimising.

## Failed / rejected

- Task-3 random-init QAT: large short-horizon deficit; retain warm start.
- Task-3 learned symmetric scale and TTQ probes: lower short-screen accuracy
  and near-zero sparsity relative to the B2 control; do not re-open without a
  specific KD×quantiser interaction hypothesis.
- T=16/lambda=0.1: short-screen leader but weaker at 30 epochs; do not select
  from early accuracy alone.

## Inconclusive

- Whether vanilla KD improves 200-epoch, three-seed validation accuracy.
- Whether CE/KD gradients conflict; Stage-4 screening did not log this signal.
- Whether the label-smoothed teacher is a limitation; no permitted alternate
  frozen teacher exists.

## Needs reproduction

- The final Task-4 T=2/lambda=0.9 result, after all three seeds finish.
- Any Task-5 candidate that clears the predefined screen gate.

## Next hypotheses

1. DKD: non-target-class relations are underweighted by coupled vanilla KD.
2. DIST: prediction relations are more attainable for the ternary student than
   exact softened probabilities.
3. Attention transfer: normalised spatial energy transfers a stable signal
   despite teacher/student depth mismatch.
4. Relational feature KD: batch geometry transfers more reliably than raw
   FP32 feature values.

## Required record fields for new experiments

Each new append-only record states `experiment_id`, `task`, `hypothesis_id`,
`parent_experiment`, `git_commit`, `config_hash`, `dataset_split_hash`,
`teacher_checkpoint_hash`, `student_initialization`, seed, method,
`changed_variables`, and `fixed_variables`. It then records validation mean/std,
best epoch, runtime, prediction/KD/gradient/representation/quantisation/stability
diagnostics, a **GREEN/YELLOW/RED** decision, mechanism evidence
(SUPPORTED/CONTRADICTED/INCONCLUSIVE), and the next action
(PROMOTE/REJECT/INVESTIGATE/REPEAT). Legacy Task 1–4 records remain immutable.
