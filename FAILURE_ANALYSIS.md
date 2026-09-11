# Failure-analysis policy

When an experiment underperforms its matched control, retain the artifact and
classify it before scheduling a successor.

| Symptom | Evidence to inspect | Targeted follow-up |
|---|---|---|
| Accuracy down, quantisation error up | layer error, scale/threshold drift, deployed codes | inspect sensitive layer; do not change KD and quantiser simultaneously |
| Accuracy down, sparsity collapses or dead layers appear | zero fraction, positive/negative balance, all-zero channels | threshold/scale diagnostic; preserve all-layer ternary requirement |
| Accuracy down, KD dominates CE | CE/KD weighted contributions, gradient norms | lower/schedule KD weight only after a matched control |
| Accuracy down, CE/KD gradients conflict | norm, cosine, conflict frequency | test a target transform or balancing mechanism, not a stack of losses |
| Stable loss but no accuracy gain | agreement, entropy, margins, teacher-correct subset | test DKD/DIST or confidence filtering |

## Final advanced-KD findings (2026-09-11)

DKD underperformed the vanilla Task 4 control: 94.88% mean versus 95.12% mean,
despite passing strict ternary and checkpoint-reload checks. DIST reached
94.94% in its one-seed screen and is therefore screening-only. These results
support the hypothesis that changing the KD decomposition does not
automatically overcome the ternary representational bottleneck. The official
test set was not used.
| Training instability | non-finite checks, LR, gradient norms, checkpoint/reload | optimisation-only smoke before changing the method |
| Apparent one-seed gain | seed distribution | reproduce; do not interpret as improvement |

The diagnosis is probabilistic.  It must label evidence as observed, inferred,
or unknown; it must never turn correlation into causal proof.

## Failure taxonomy

| ID | Failure class | Required evidence | Consequence |
|---|---|---|---|
| F1 | accuracy failure | matched validation delta and seed context | weaken/reject only after reproduction gate |
| F2 | optimisation failure | non-finite loss, gradient/learning-rate trace | stabilise before comparing methods |
| F3 | KD-signal failure | KD contribution and intended prediction signal unchanged | weaken KD hypothesis |
| F4 | CE/KD gradient conflict | measured norm ratio/cosine/conflict frequency | test targeted balancing/target change only |
| F5 | representation mismatch | feature/attention/relation diagnostic | alter only the transfer target |
| F6 | quantisation bottleneck | layer error/sparsity/scale/dead-layer evidence | inspect quantiser path separately |
| F7 | ternary instability | deployed-code/coverage or scale/threshold instability | invalid until repaired |
| F8 | overfitting | training/validation gap and trajectory | regularisation/schedule hypothesis |
| F9 | hyperparameter sensitivity | controlled local perturbation | constrain the search family |
| F10 | implementation/configuration error | invariant, reload, or provenance failure | invalidate and repair |
| F11 | experiment contamination | declared/fixed-variable comparison | not comparable to parent |
| F12 | insufficient statistical evidence | effect below seed variation or too few seeds | YELLOW; repeat rather than interpret |

Every failure record includes symptom, observed diagnostics, likely mechanism,
confidence, next experiment, and whether its hypothesis is weakened, falsified,
or unresolved.
