# Research experiment memory

## CURRENT RESEARCH EXECUTION STATE

Current mode is **TIME-CONSTRAINED UNIFIED NOTEBOOK EXECUTION**. Task 4 has
one preserved completed seed (42) and two remaining seeds (43, 44). Deep
AutoResearch/AutoML is **PAUSED, NOT ABANDONED**; this database and the
associated research documents are the explicit future resumption checkpoint.

`experiments.jsonl` is append-only.  Every line conforms to `schema.json` and
records an experiment or an explicit pending state.  Training scripts should
write their normal result JSON first; the registry then references that
immutable artifact rather than duplicating checkpoints.

Before proposing a run, inspect prior entries with the same parent control,
method, and factor values.  Do not repeat a completed experiment unless the
new entry declares `purpose: reproduction`.

## Traceability model

`Hypothesis → experiment → configuration → result → diagnostics → decision →
next hypothesis/experiment`.

The ledger must answer why a run was started, which evidence promoted or
rejected it, and which hypotheses remain unresolved. New non-legacy entries
declare parent control, hypothesis ID, mechanism, changed/fixed variables, and
provenance hashes. Run `validate_experiment_contamination.py` before a record
is accepted as a clean comparison.
