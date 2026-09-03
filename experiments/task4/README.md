# Task 4 evidence layout

This directory contains validation-only evidence for Task 4.  It is kept separate from the immutable Task 3 B2-default baseline.

- `audit/`: protocol and checkpoint provenance snapshots.
- `smoke/`: loss, freezing, ternary-coverage, and end-to-end smoke evidence.
- `t_lambda/`: staged temperature/lambda grid rankings and stdout logs.
- `teacher/`: only pre-existing, frozen-teacher comparison runs.
- `curriculum/`: optional fixed-schedule diagnostics, labelled non-canonical.
- `optimization/`: validation-only optimizer and LR diagnostics.
- `final/`: three-seed 200-epoch selected-condition runs.
- `diagnostics/`, `checkpoints/`, and `histories/`: run-addressable artifacts written by the production trainer.

No path in this tree may contain a test-set evaluation.
