# Research experiment memory

## CURRENT RESEARCH EXECUTION STATE

Current mode is **TIME-CONSTRAINED FINALIZATION**. Task 4 is complete for
seeds 42, 43, and 44 with a 95.12% +/- 0.15% validation control. Deep
AutoResearch/AutoML is **PAUSED, NOT ABANDONED**; this database and the
associated research documents are the explicit future resumption checkpoint.

`experiments.jsonl` is append-only.  Every line conforms to `schema.json` and
records an experiment or an explicit pending state.  Training scripts should
write their normal result JSON first; the registry then references that
immutable artifact rather than duplicating checkpoints.

Before proposing a run, inspect prior entries with the same parent control,
method, and factor values.  Do not repeat a completed experiment unless the
new entry declares `purpose: reproduction`.

The final artifact registry and report are in `report/final/`. DKD is recorded
as rejected at 94.88% mean, DIST as screening-only at 94.94%, and the official
test set remains locked.

## Traceability model

`Hypothesis → experiment → configuration → result → diagnostics → decision →
next hypothesis/experiment`.

The ledger must answer why a run was started, which evidence promoted or
rejected it, and which hypotheses remain unresolved. New non-legacy entries
declare parent control, hypothesis ID, mechanism, changed/fixed variables, and
provenance hashes. Run `validate_experiment_contamination.py` before a record
is accepted as a clean comparison.
