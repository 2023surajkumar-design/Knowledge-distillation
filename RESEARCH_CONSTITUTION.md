# ATDL Research Constitution

## CURRENT RESEARCH EXECUTION STATE

**Mode: TIME-CONSTRAINED FINALIZATION.** Task 4 is complete for seeds 42, 43,
and 44 with a 95.12% +/- 0.15% validation control. Preserve all Task 1–4
artifacts and do not overwrite or reinterpret them. Broad AutoResearch/AutoML
is **PAUSED, NOT ABANDONED**, and must remain resumable from the research
documents and `research_db/`. The official test set remains locked.

This constitution governs every experiment after Task 4.  It supersedes earlier
forward-looking plans when they conflict with the experimentally established
Tasks 1–4 or this test-set firewall.

## Immutable scientific controls

- **Data and split:** CIFAR-10 only; the frozen classwise 45k/5k training/
  validation split in `src/data.py`; no external data.
- **Test firewall:** no Task 5–9 script may import or call `get_test_loader`.
  Test evaluation is a single, separately authorized final action after all
  method choices have been frozen.
- **Models:** the frozen Task-1 FP32 CIFAR ResNet34 teacher and CIFAR ResNet18
  student.  No architecture enlargement, replacement, or extra inference
  module is allowed.
- **Deployment constraint:** every ResNet18 convolution and FC deployed weight
  is ternary; latent FP weights are allowed only for QAT optimisation.  Any
  training-only connector/projection must be absent from the deployed model.
- **QAT and teacher safety:** ternary forward passes are active at every Task
  3+ optimisation step.  Teachers are `eval()`, non-trainable, and executed
  under `torch.no_grad()`.
- **Evidence preservation:** previous checkpoints, histories, results, and
  Task 3/4 code paths are append-only.  Every run has a unique name and may
  not overwrite an existing artifact.

## Research freedom, with evidence gates

Allowed factors are KD formulation/weighting/targets, training-only feature
transfer, quantizer thresholds/scales/STE, optimiser and schedule, EMA,
gradient clipping, and curricula.  A factor can advance only when its parent
control is fixed, its hypothesis is recorded, its validation-only result is
stored, and its diagnostics explain at least a plausible mechanism.

## Scientific governance additions

- **Accuracy is necessary, not sufficient.** A validation gain without a
  diagnostic/stability account is preliminary.
- **Screening is not confirmation.** One-seed short-budget results are signals
  for promotion decisions, never headline findings.
- **Negative results remain evidence.** A failed or inconclusive method is
  retained in the ledger with its hypothesis and failure classification.
- **Mechanism claims require matching measurements.** No gradient-conflict
  explanation is permitted without measured CE/KD gradient geometry.
- **Explore aggressively; advance conservatively.** Research freedom is broad,
  but promotion must be evidence-based.

## Contamination rule

Every new experiment declares `parent_control`, `hypothesis_id`, `mechanism`,
`changed_variables`, and `fixed_variables`. Dataset/split, teacher, student
architecture/initialisation policy, QAT baseline, quantiser, seed protocol, and
budget are protected unless an explicit hypothesis says otherwise. An
undeclared mismatch is **CONTAMINATED** and is not clean evidence against its
parent. `scripts/validate_experiment_contamination.py` performs the pre-ledger
check.

## Decision rules

1. Search one causal idea at a time; do not combine unproven components.
2. Use one seed for screening, then two seeds for confirmation, then three
   independent seeds for a headline claim.
3. Rank candidates using validation mean, uncertainty, ternary correctness,
   sparsity/quantisation metrics, and stability—not a best lucky seed.
4. Any apparent gain smaller than observed seed noise is **inconclusive**,
   not a method improvement.
5. A failed experiment is evidence.  It must receive a mechanism-oriented
   diagnosis before a follow-up is scheduled.
6. Never tune on the test set or use it to choose a method, hyperparameter,
   or checkpoint.

## Required evidence per experiment

Each result must have an immutable config, parent control, seed/budget, model
and teacher provenance, validation/loss/KD/quantisation/optimisation metrics,
runtime, interpretation, decision, and next action in `research_db/`.

## Current boundary

The Task 4 final three-seed run is incomplete: seed 42 completed, seed 43 was
interrupted after an append-only best checkpoint, and seed 44 did not start.
Its results remain provisional until a separately named, non-overwriting
completion/reproduction produces all per-seed summaries. No Task 5 training may
start before `POST_TASK4_AUDIT.md` is updated with that final evidence.
