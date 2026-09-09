# Research hypotheses registry

Every post–Task-4 experiment must cite one primary hypothesis ID. A status is
updated only from append-only evidence in `research_db/`; it is not inferred
from a promising single screen.

## H1: DKD isolates transferable non-target knowledge

**Status:** UNTESTED  
**Rationale:** coupled vanilla KL may underweight non-target relations; DKD
separates target- and non-target-class components.  
**Literature motivation:** Zhao et al., CVPR 2022.  
**Independent variable:** DKD objective and its declared TCKD/NCKD coefficients.  
**Dependent variables:** validation accuracy, agreement, margins, CE/KD gradient
cosine, gradient ratio, stability.  
**Expected evidence:** a matched, reproducible improvement with a changed
non-target diagnostic and no unacceptable quantisation regression.  
**Falsification:** no Gate-C reproducible advantage over matched vanilla KD, or
the intended non-target signal does not change.  
**Linked experiments:** T5-DKD screen → confirmation.

## H2: DIST prediction relations are more attainable than exact KL

**Status:** UNTESTED  
**Rationale:** a strict ternary student may preserve prediction relations even
when it cannot reproduce an FP32 teacher probability vector.  
**Literature motivation:** Huang et al., NeurIPS 2022.  
**Independent variable:** DIST relation loss with fixed parent recipe.  
**Dependent variables:** validation accuracy, prediction-relation score,
agreement, entropy, CE/KD gradient alignment.  
**Expected evidence:** improved relation agreement plus reproducible accuracy or
stability benefit.  
**Falsification:** no Gate-C gain and no intended prediction-relation change.  
**Linked experiments:** T5-DIST screen → confirmation.

## H3: normalised attention transfer preserves useful spatial information

**Status:** UNTESTED  
**Rationale:** normalised spatial energy may be less sensitive to FP32 feature
scale and teacher/student channel mismatch than direct feature regression.  
**Literature motivation:** Zagoruyko and Komodakis, ICLR 2017.  
**Independent variable:** one training-only late-stage attention-transfer loss.  
**Dependent variables:** validation accuracy, attention alignment,
representation cosine, feature norms, quantisation metrics.  
**Expected evidence:** attention alignment changes and survives Gate C without
changing deployed architecture.  
**Falsification:** alignment improves without accuracy/stability benefit, or
alignment itself does not improve.  
**Linked experiments:** T5-AT screen → confirmation.

## H4: relational feature transfer preserves useful geometry

**Status:** UNTESTED  
**Rationale:** pairwise feature geometry may survive the ternary bottleneck more
readily than pointwise FP32 activations.  
**Literature motivation:** Park et al., CVPR 2019.  
**Independent variable:** one late-stage cosine or distance/angle relation loss.  
**Dependent variables:** validation accuracy, relation agreement,
class-centroid similarity, gradient alignment, runtime.  
**Expected evidence:** relational alignment and a replicated validation benefit.  
**Falsification:** no Gate-C benefit or unstable batch-relation gradients.  
**Linked experiments:** T5-RKD screen → confirmation.

## H5: KD benefit is associated with quantisation difficulty

**Status:** UNTESTED  
**Rationale:** lower Task-4 aggregate quantisation error is a clue, but not proof,
that quantisation difficulty may modulate usable supervision.  
**Literature motivation:** quantisation-aware distillation/QFD literature.  
**Independent variable:** initially none; observational layer/sample diagnostic,
then one declared Task-6 weighting intervention.  
**Dependent variables:** Pearson/Spearman association between error, feature
distortion, KD gradient magnitude, disagreement, and benefit.  
**Expected evidence:** a stable association across independent runs.  
**Falsification:** no meaningful repeatable relationship.  
**Linked experiments:** T6 diagnostic baseline → one-factor Task-6 test.

## H6: disagreement identifies where KD is useful

**Status:** UNTESTED  
**Rationale:** teacher/student disagreement may identify examples where teacher
targets are useful—or where they are unattainable.  
**Literature motivation:** student-aware/filtered KD research direction.  
**Independent variable:** initially diagnostic grouping by teacher correctness,
confidence, and disagreement.  
**Dependent variables:** subgroup accuracy, entropy, KD loss, gradient magnitude.  
**Expected evidence:** consistent subgroup pattern across seeds.  
**Falsification:** no reproducible subgroup relationship.  
**Linked experiments:** T6 prediction diagnostic; confidence filter only if promoted.

## H7: CE/KD gradient conflict explains weak vanilla-KD benefit

**Status:** UNTESTED  
**Rationale:** accuracy and loss curves cannot reveal objective conflict.  
**Literature motivation:** multi-objective optimisation and STE diagnostics.  
**Independent variable:** measured CE/KD gradient geometry across matched runs.  
**Dependent variables:** CE/KD norm ratio, cosine, conflict frequency, accuracy.  
**Expected evidence:** consistently negative cosine or harmful ratio co-occurring
with weak/negative KD effect.  
**Falsification:** mostly positive/near-orthogonal cosine with no conflict trend.  
**Linked experiments:** T5.0 diagnostic vanilla control; candidate screens.

## H8: representation alignment and accuracy are not equivalent

**Status:** UNTESTED  
**Rationale:** a feature signal may improve cosine alignment but fail to improve
classification under a discrete weight constraint.  
**Literature motivation:** feature/attention/relational KD.  
**Independent variable:** T5 feature method versus matched vanilla.  
**Dependent variables:** feature cosine, centroid similarity, accuracy, stability.  
**Expected evidence:** explicitly report agreement or dissociation.  
**Falsification:** not applicable as a structural observation; it is supported
only when a dissociation is observed.  
**Linked experiments:** T5-AT and T5-RKD.

## H9: best ternary KD may differ from conventional FP32 KD

**Status:** UNTESTED  
**Rationale:** strict ternary representational limits change the attainable
target set and may alter the best KD objective.  
**Literature motivation:** DIST strong-teacher setting and quantisation KD.  
**Independent variable:** controlled method identity at fixed QAT recipe.  
**Dependent variables:** validation accuracy, error/sparsity, objective geometry.  
**Expected evidence:** one method beats matched vanilla and T3 robustly.  
**Falsification:** no advanced candidate has repeatable benefit.  
**Linked experiments:** full Task-5 ladder.

## H10: optimise only after identifying transfer mechanism

**Status:** UNTESTED  
**Rationale:** changing KD, STE, quantiser, and schedule together destroys
attribution and validation-budget discipline.  
**Literature motivation:** hierarchical HPO practice.  
**Independent variable:** approval state of an experiment family.  
**Dependent variables:** comparability, seed stability, reproducibility.  
**Expected evidence:** only Gate-D mechanism finalists enter constrained AutoML.  
**Falsification:** a deliberate authorised joint study would require its own
controls; absence of a candidate does not prove the principle false.  
**Linked experiments:** Task-9 AutoML gate.
