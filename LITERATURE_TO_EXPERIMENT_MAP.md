# Literature-to-experiment map

This map records why a method is considered, what assumption may fail under
strict ternary QAT, and the smallest controlled test.  A citation motivates a
hypothesis; it is not evidence that the method will work here.

| Paper / method | Hypothesis | Knowledge transferred / ternary rationale | Failure mode + required diagnostic | Experiment / evidence status |
|---|---|---|---|---|
| [Zhao et al., CVPR 2022 — DKD](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html) | H1 | Target/non-target logits; independent weighting may retain transferable non-target relations. | Coefficient overfit; prediction diagnostics plus CE/KD gradient geometry. | T5-DKD / UNTESTED |
| [Huang et al., NeurIPS 2022 — DIST](https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html) | H2 | Inter-/intra-class prediction correlation may be attainable without exact KL matching. | Noisy batch relations; relation score, agreement, and relation-gradient stability. | T5-DIST / UNTESTED |
| [Zagoruyko & Komodakis, ICLR 2017 — attention transfer](https://openreview.net/pdf?id=Sks9_ajex) | H3 | Normalised spatial energy; dimension-free scale-normalised target may survive ternarisation. | Stage mismatch; attention alignment, feature norm, and accuracy dissociation. | T5-AT / UNTESTED |
| [Park et al., CVPR 2019 — RKD](https://arxiv.org/abs/1904.05068) | H4 | Pairwise feature geometry; relations may survive where values do not. | Batch sensitivity; class-centroid/relation alignment and runtime. | T5-RKD / UNTESTED |
| [Zhu et al., AAAI 2023 — QFD](https://ojs.aaai.org/index.php/AAAI/article/view/26354) | H5 | Quantisation-friendly features; possible Task-6 student-aware target. | Original quantized-teacher assumption fails here; layer-error association first. | T6 diagnostic / DEFERRED |
| [Yin et al., ICLR 2019 — STE](https://iclr.cc/virtual/2019/poster/671) | H7/H10 | Better coarse-gradient geometry may improve stable QAT. | Scope differs from weight ternarisation; gradient/scale/stability metrics. | T7 conditional / DEFERRED |

## Explicitly deferred proposals

Contrastive KD, raw feature MSE/FitNet, cross-layer learned connectors, dynamic
temperature, sample-wise lambda, progressive QAT, and large joint HPO are
deferred.  They increase degrees of freedom before there is evidence that the
smaller logit/attention/relation hypotheses fail.  This is experimental
discipline, not a claim that the methods are ineffective.
