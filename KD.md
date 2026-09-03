# KD.md — Master Research & Execution Plan
## Knowledge Distillation with Ternary-Weight QAT (ResNet34 → Ternary ResNet18, CIFAR-10)

> **ATDL Assignment I — 30 marks — Deadline: Wed 16 Sep 2026, 23:59.** Today is 28 Aug 2026 → **~19 days**.
> This is the single source of truth for the whole project. It is a *plan*, not an implementation. We execute **one task at a time**; nothing below authorizes jumping ahead. When a task is invoked ("WORK ON TASK N"), we implement only that task while keeping every interface compatible with later tasks.

---

## 0. How to read this document

- **Section 1** — fixed project context: the pipeline, the hard constraints vs. research freedom, the rubric, and the exact status of Task 1 (done).
- **Section 2** — the tooling layer (Karpathy `autoresearch` + Optuna/AutoML), what each *actually* is (verified), and — critically — how to use them **without violating the assignment's AI-disclosure policy**.
- **Section 3** — the task-by-task deep breakdown (Tasks 2→12), each with role, hypothesis, sub-tasks, the full method design space, hyperparameters + search ranges, how the agent/HPO applies, deliverables, ablations, rubric linkage, and pitfalls.
- **Section 4** — the central research thesis (knowledge transformation under a discrete capacity constraint) that ties the advanced tasks together.
- **Sections 5–10** — master hyperparameter tables, the controlled experiment ladder, compute/schedule to the deadline, repo structure, reproducibility, risk register, and the honest AI-usage/novelty statement.

A ⚑ marks a place where a common default is wrong for *this* problem and we deliberately deviate.

---

## 1. Fixed Project Context

### 1.1 The pipeline

```
CIFAR-10 (50k train / 10k test, 32×32×3, 10 classes)
        │
        ▼
  ResNet34  FP32  TEACHER   ──(Task 1: DONE, 95.19% test)──┐
        │  frozen, teacher.eval(), no grad leakage         │
        ▼                                                   │
  ResNet18  STUDENT                                         │
        │   weights ∈ {−α, 0, +α}  (conv + FC)              │
        ▼                                                   │
  QAT (ternary forward every step) + KD (from frozen teacher)
        │
        ▼
  compressed ternary student  → evaluate vs teacher & vs FP32 ResNet18-no-KD
```

### 1.2 Hard constraints (from the assignment — never violate)

| # | Constraint |
|---|---|
| C1 | Dataset = CIFAR-10 only. No extra data, no external images, test set touched **once** at the very end. |
| C2 | Teacher = ResNet34, FP32, trained from scratch. **Do not change its architecture.** |
| C3 | Student = ResNet18. **Do not change its architecture; do not use a larger student.** |
| C4 | Student conv **and** FC weights must be ternary `{−α, 0, +α}` at inference. α may be per-tensor / per-layer / per-channel. |
| C5 | Ternary constraint enforced **during** training (QAT), not post-hoc PTQ. Latent FP weights kept separately from the quantized forward weights. |
| C6 | Teacher frozen during student training; no gradient into teacher. |
| C7 | Required comparisons: (a) FP32 ResNet34 teacher, (b) FP32 ResNet18 no-KD baseline, (c) ternary ResNet18 KD+QAT. |
| C8 | Report accuracy, model size, sparsity, compression ratio, accuracy degradation. |
| C9 | ≥1 meaningful ablation on **T**, **λ**, or **Δ/threshold**. |
| C10 | Reproducibility: fixed seeds, requirements/env file, documented code, checkpoints, ternary-verification script. |

### 1.3 Research freedom (everything else is a design space to investigate)

Optimizer, LR schedule, augmentation, regularization, normalization, KD formulation (logit/feature/relation/attention/contrastive), temperature policy, KD weighting policy, ternarization threshold/scale formulation, STE variant, QAT schedule (progressive/curriculum), warmup, EMA, gradient-conflict handling, adaptive/sample-wise KD, layer-sensitivity weighting, consistency losses, quantization-error-aware losses.

### 1.4 Rubric → where each mark is earned (30 total)

| Rubric component | Marks | Which tasks earn it |
|---|---|---|
| Ternary quantizer & student architecture (α, Δ; latent≠forward; STE; which layers FP32 documented) | 6 | **T3, T7** |
| KD formulation & integration (T-scaled KL + CE; teacher frozen; T,λ explored) | 6 | **T4, T5, T9** |
| Joint KD+QAT pipeline & stability (ternary forward consistent; stable curves; QAT-sensible LR/opt; debugging evidence e.g. progressive/warmup) | 7 | **T4, T6, T8, T10** |
| Evaluation & compression analysis (teacher vs FP32-18 vs ternary; size/sparsity/compression correct; honest accuracy-loss discussion) | 5 | **T12** |
| Report quality, ablation & critical discussion (citations; ≥1 ablation; limitations incl. real-HW-speedup-needs-kernels, early/late-layer impact) | 6 | **T11 + report** |

> ⚑ **The single highest-value row is the 7-mark "pipeline & stability" one.** It rewards a *working, stable, debugged* ternary-KD-QAT run more than it rewards exotic methods. Priorities follow this: get a stable T3→T4 pipeline before any advanced-KD research.

### 1.5 Task 1 status — DONE, with two carry-forward notes

**Result (executed on RTX 4080 SUPER):** CIFAR-adapted ResNet34 (3×3 stride-1 stem, no max-pool, `[3,4,6,3]`, 21,282,122 params, FP32), stratified 45k/5k split, test evaluated once → **95.19% test** (best val 95.54% @ epoch 190), train ~100%, train–val gap ~4.6pp, 55 min. Checkpoint + history saved and verified to reload identically. This is squarely in the well-trained-plain-ResNet range and comfortably above the original ResNet paper's CIFAR numbers (ResNet-110 = 93.57%).

**Carry-forward note A — the teacher was trained with `label_smoothing=0.1`.** ⚑ Müller, Kornblith & Hinton (*"When Does Label Smoothing Help?"*, NeurIPS 2019) show label smoothing improves a teacher's own accuracy but **erases inter-class relational structure in the penultimate layer — the exact "dark knowledge" KD depends on — making a label-smoothed teacher a *worse* teacher.** Shen et al. (ICLR 2021) qualify this (one-sided/contextual, sometimes still net-positive). **Action for T4:** keep the LS=0.1 teacher **and** train a second identical teacher with **LS=0.0**, then distill from both and compare. Cost: one extra ~1h run. This is a cheap, rubric-rewarded, research-grade control that de-risks the entire KD stage. *(This is prepared in T1's scope but only *used* from T4 onward — no premature work.)*

**Carry-forward note B — figures were not captured** in the executed T1 notebook (headless Agg backend; 0 images embedded). Add `%matplotlib inline` / `savefig` and re-run so the report has the curves. Also add `weights_only=False` to every future `torch.load` of the teacher (PyTorch ≥2.6 default flipped).

---

## 2. Tooling Layer — AutoResearch + AutoML (verified, with honest framing)

### 2.1 What `autoresearch` actually is (verified 2026-08-28)

- **Real repo:** `github.com/karpathy/autoresearch`, MIT, released ~Mar 2026. Tagline: *"AI agents running research on single-GPU nanochat training automatically."* It is a stripped one-file derivative of nanochat.
- **Loop:** an external coding agent (Claude Code / Codex / etc., run with permissions disabled) edits **one file** `train.py`, runs a **fixed ~5-minute** training, checks the metric, and **keeps the git commit only if the metric improved** (a "ratchet"; failures reverted). ~12 experiments/hour.
- **Files:** `prepare.py` (data + eval, **immutable**), `train.py` (**the only file the agent edits**), `program.md` (human-written instructions/guardrails), `pyproject.toml`. Metric in the original is `val_bpb` (bits-per-byte, LM task).
- **Setup:** `uv sync → uv run prepare.py → uv run train.py`; Python 3.10+, one NVIDIA GPU.
- ⚑ **Correction to our `AutoMLresearch.md`:** the "Guillaume Erhard, ResNet-20, 91.9%→95.4%, 5-min, architecture edits" story is **unverified — no such write-up was found.** The real, verifiable CIFAR-10 port is by **"L.J." (Medium)**: **90.12%→94.55%**, **~1-minute** trials, and the agent's edits were mostly **throughput/efficiency** (in-place ReLU, `torch.compile`, GPU preloading, AMP) — *not* schedule/aug/architecture. Do **not** cite the Erhard numbers in the report.
- ⚑ **Karpathy's own stated limits:** the ratchet gets **stuck in local search** ("cycles minor variations of whatever worked last"), can't take temporarily-worse steps, and is **automated recipe/hyperparameter tuning — not autonomous novel research.** The 5-min window is wall-clock and **hardware-specific** (a config found on one GPU won't transfer).

### 2.2 What the AutoML/HPO layer (Optuna) gives us — and why it beats the agent for numbers

- Peer-reviewed finding ("Can LLMs Beat Classical HPO? A Study on autoresearch", arXiv 2603.24647): **within a fixed numeric search space, classical HPO (TPE, CMA-ES) consistently beats LLM agents.** The agent's edge is **structural/code** changes (loss form, quantizer, OOM/NaN guards), not tuning numbers. Their best result is a **hybrid ("Centaur")**: classical optimizer owns the numbers, LLM proposes code edits on a minority of trials.
- **Recommended stack (single GPU):** **Optuna** with **`TPESampler(seed=SEED)` + `HyperbandPruner`** (Optuna's own documented best pairing) — multi-fidelity early stopping is the single biggest results-per-GPU-hour lever when each run is 30–60 min. Drive it from a plain loop (or PyTorch Lightning + `PyTorchLightningPruningCallback`). Persist the study in **SQLite** (resumable). Log every trial to **W&B or MLflow**. Avoid NNI (archived Sep 2024), HpBandSter (dead), DEHB (frozen). Ray Tune only if we add GPUs; Ax/SMAC3 only if we want Bayesian/multi-objective (e.g., accuracy-vs-size).

### 2.3 Division of labor (our operating model)

| Layer | Owns | Examples in this project |
|---|---|---|
| **Human (you) + this plan** | Hypotheses, constraints, which method to try, interpretation | Choosing DKD vs DIST vs QFD; deciding to test LS=0 teacher |
| **`autoresearch` agent** | *Code/structure* of `train.py` under strict `program.md` guardrails | STE variant implementation, quantizer code, augmentation pipeline, LR-schedule *shape*, AMP/throughput, OOM/NaN guards |
| **Optuna (TPE+Hyperband)** | *Numeric* hyperparameters over a fixed, working `train.py` | T, λ, LR, weight decay, threshold factor t, feature-loss weight γ |

> ⚑ **Never let the agent tune numbers that Optuna should own, and never let either change anything under §1.2.** `program.md` must forbid: architecture edits, extra data, touching the test set, un-freezing the teacher, or relaxing the ternary constraint.

### 2.4 ⚑ AI-disclosure policy — non-negotiable for this assignment

The assignment: *"dumping all the problem statement… to get the solution and then submitting… will give you zero points. You have to attach the conversation thread you used… If your code and AI chats differ, there will be penalty."* Therefore:

1. **Keep and attach the AI chat threads** (this planning thread, any autoresearch agent transcripts, any Optuna-assisted coding chats). The `autoresearch` git history *is* an experiment log — keep it; it also documents what the agent changed.
2. **Frame autoresearch honestly in the report**: "automated hyperparameter/recipe search within fixed architecture and dataset constraints," **not** "AI did the research." This matches the tool's real capability and the L.J./Karpathy framing.
3. **You must understand and be able to explain every line** the agent writes. Where the agent proposes a method, the *hypothesis and justification* in the report must be yours (Section 3 gives them).
4. **Do the novelty framing in Section 10** ("to the best of our literature search…") — never claim unqualified novelty.

### 2.5 Repo structure (create once, before Task 2)

```
atdl-kd-ternary/
  README.md                      # end-to-end run instructions (rubric requires "runs end-to-end")
  requirements.txt / pyproject.toml
  configs/                       # one YAML per experiment (seed, all HPs)
  data/                          # CIFAR-10 (gitignored)
  src/
    data.py                      # CIFAR-10 pipeline + stratified 45k/5k split (frozen across all tasks)
    models/resnet_cifar.py       # ResNet34 + ResNet18 CIFAR-adapted (shared defs)
    quant/ternary.py             # TernaryConv2d/Linear, quantizer, STE variants
    kd/losses.py                 # vanilla KD, DKD, DIST, feature/relation losses
    train_teacher.py             # Task 1 (done)
    train_student_fp32.py        # Task 2
    train_student_ternary.py     # Task 3/4/5/6 (flag-driven)
    eval/verify_ternary.py       # C10 verification script
    eval/compression.py          # size/sparsity/compression (Task 12)
  autoresearch/                  # forked karpathy/autoresearch per stage
    prepare.py program.md train.py pyproject.toml
  hpo/                           # Optuna studies (*.db) + sweep scripts
  checkpoints/  results/  plots/  report/
```

- ⚑ `src/data.py` (the split) is written **once** and **frozen** — every task imports the identical split so comparisons are controlled (C1, experimental discipline). Mirror it into each `autoresearch/prepare.py` as the immutable file.
- Never overwrite results;每 experiment writes a new timestamped/config-named artifact.

### 2.6 Reproducibility & tracking (applies to all tasks)

Seed torch/numpy/random per run; derive each Optuna trial's seed from `trial.number`; seed the sampler; log git SHA + library/CUDA versions per run (QAT/STE are version-sensitive); store Optuna study in SQLite; save best checkpoint + full config + training history JSON per experiment; run 3 seeds for any headline conclusion and report mean ± std.

---

## 3. Task-by-Task Deep Plan

> Legend per task: **Role** · **Hypothesis** · **Deep sub-tasks** · **Design space (advanced choices)** · **Hyperparameters & ranges** · **Agent/HPO strategy** · **Deliverables** · **Ablations** · **Rubric** · **Pitfalls**.

---

### TASK 2 — FP32 ResNet18 baseline (the student's intrinsic capacity)

- **Role.** The lower reference for "what ResNet18 can do unconstrained," and the required no-KD baseline (C7b). Also produces the **FP32 student init** that Task 3/4 can warm-start from.
- **Hypothesis.** ResNet18 FP32 lands ~1–1.5pp below the ResNet34 teacher on CIFAR-10; this gap is the *capacity* component, separate from the *quantization* component we measure later.
- **Deep sub-tasks.**
  1. CIFAR-adapted ResNet18 (`[2,2,2,2]` BasicBlocks, 3×3 stride-1 stem, no max-pool, FC→10). Sanity: ~11.17M params, `[B,10]` output, FP32.
  2. Reuse the *exact* T1 recipe (SGD+Nesterov, wd 5e-4, warmup+cosine, LS 0.1) and the frozen split → clean, comparable baseline.
  3. Train 3 seeds; save best on val; test once; save checkpoint + history + curves.
  4. (Prepared, not run here) a **`--kd` flag** wiring for later FP32-student+KD sanity — implemented but left off in T2.
- **Design space (advanced choices).** Augmentation strength is the main lever (Cutout / RandAugment / Mixup / CutMix can add ~0.5–1.5pp each); EMA of weights (+0.2–0.5pp); zero-init residual BN-γ ("Bag of Tricks", +~0.4pp). ⚑ Keep the *baseline* clean (crop+flip only) so it's a fair reference; report augmentation upgrades as a separate labeled config.
- **Hyperparameters & ranges** (Optuna optional here — the recipe is already strong): LR log-uniform 3e-2…2e-1; wd 1e-4…1e-3; label smoothing {0.0, 0.1}; epochs 150–200.
- **Agent/HPO strategy.** Low priority for automation; a 2–3 trial manual/Optuna check suffices. If you want to exercise the autoresearch loop *once* on an easy target to learn the workflow, this is the safe place (it cannot break constraints on a plain FP32 net).
- **Deliverables.** `resnet18_fp32_best.pth`, history JSON, curves, 3-seed mean±std test acc.
- **Ablations.** (Optional) crop+flip vs +Cutout vs +Mixup — informs which augmentation to carry into ternary training.
- **Rubric.** Feeds Evaluation table (5). 
- **Pitfalls.** Don't accidentally tune on the test set; keep the split identical to T1.

---

### TASK 3 — Ternary ResNet18 QAT baseline (no KD)

- **Role.** Isolates the pure **cost of ternarization** (FP32-18 → ternary-18, no teacher). This is the reference that reveals whether KD later *helps or hurts* (the central empirical question).
- **Hypothesis.** A correctly-implemented ternary QAT ResNet18 with learned per-channel scales retains most of FP32-18's accuracy (TTQ-style results suggest small drops on CIFAR); the residual gap localizes to sensitive layers (esp. `conv1`).
- **Deep sub-tasks.**
  1. **`TernaryConv2d` / `TernaryLinear`** holding a **latent FP32 weight** `W`; forward uses `Ŵ = α · Q(W)`, `Q(W)∈{−1,0,+1}`. ⚑ Latent `W` and forward `Ŵ` must be cleanly separated (rubric-explicit).
  2. **Quantizer.** Start TTQ-style: threshold `Δ_l = t·max(|W_l|)` (t≈0.05), learned per-**channel** scales `α_p, α_n` (asymmetric) — a permitted, stronger form of `{−α,0,+α}`. Provide TWN closed-form (`Δ=0.7·E|W|`, single α) as a comparison.
  3. **STE (backward).** Start with clipped-STE (identity inside `[−1,1]`, 0 outside — the variant Yin et al. prove has positive gradient correlation). Provide a hook to swap variants (Task 7).
  4. **Which layers stay FP32?** ⚑ Document the choice. Common practice keeps the first conv and final FC in higher precision; the assignment wants **both conv and FC ternary**, so default = **ternarize everything**, but *measure* first-conv/FC sensitivity (Task 13-style probe) and, if needed, compensate via finer per-channel scale / lower threshold on those layers — **without exempting them from ternary** (respects C4).
  5. Sanity gates before long runs: (a) after a step, unique forward-weight values ⊆ `{−α,0,+α}` per channel; (b) gradients flow to latent `W`; (c) loss not NaN; (d) 1-epoch smoke test.
  6. Train 3 seeds; save best on val; verify ternary constraint programmatically.
- **Design space (advanced choices).** Per-tensor vs per-layer vs per-channel α (per-channel usually best); symmetric vs asymmetric (`α_p≠α_n`, TTQ); learned vs heuristic threshold; init from **FP32-18 (Task 2)** vs from scratch (⚑ TTQ initializes from a pretrained FP model — warm-start almost always helps QAT stability); weight-decay applied to latent weights only.
- **Hyperparameters & ranges.** LR 1e-2…1e-1 (log); wd 1e-5…5e-4; threshold factor t 0.02…0.1; α init strategy {mean|abs|learned}; epochs 150–200; ⚑ QAT is LR-sensitive because STE gradients are approximate — expect a *lower* sweet-spot LR than FP32.
- **Agent/HPO strategy.** **Optuna owns** {LR, wd, t}. **Agent owns** the STE/quantizer *code* variants and OOM/NaN guards (per §2.3). Prune with Hyperband but ⚑ set min-epochs high enough (~15–20) that noisy early QAT accuracy doesn't trigger false culls.
- **Deliverables.** `resnet18_ternary_noKD_best.pth`, `verify_ternary.py` output (sparsity, α per layer), curves, 3-seed mean±std.
- **Ablations.** per-layer vs per-channel α; symmetric vs asymmetric; learned vs 0.7·E|W| threshold; warm-start vs scratch. (These also feed the required ablation set.)
- **Rubric.** Ternary quantizer & architecture (6); contributes to pipeline stability (7).
- **Pitfalls.** ⚑ Accidentally training FP32 then quantizing at the end (that's PTQ — forbidden by C5). Ternary forward must be active **every** step. Watch for dead layers (all-zero after thresholding) if t too high.

---

### TASK 4 — Vanilla KD + ternary QAT (the pivotal comparison)

- **Role.** Establishes the conventional KD baseline and answers the project's first real research question: **does vanilla KD help or hurt the ternary student vs Task 3?**
- **Hypothesis (held with genuine uncertainty).** Vanilla logit KD *may* help, but recent low-bit evidence (SQAKD; QFD) shows naive KD can **underperform no-KD** at very low precision. ⚑ Report the T4-vs-T3 delta prominently **whatever its sign** — a demonstrated "KD hurts here" is a valid, rubric-rewarded finding, not a failure.
- **Deep sub-tasks.**
  1. Load frozen teacher(s): the LS=0.1 T1 teacher **and** the LS=0.0 teacher (carry-forward note A). `teacher.eval()`, `requires_grad=False`, assert no teacher grad.
  2. Loss: `L = (1−λ)·CE(z_s, y) + λ·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T))`. ⚑ Keep the `T²` factor or λ and T become entangled.
  3. Train ternary student (from T3 config) with KD; 3 seeds each for the chosen (T,λ).
  4. Compare: T3 (no KD) vs T4 (vanilla KD) vs both teachers.
- **Design space.** Teacher choice (LS=0 vs LS=0.1) — a rubric-friendly control; KD on logits only (here) vs +feature (T5); hard-label CE vs KD-only (SQAKD-style label-free) as an ablation.
- **Hyperparameters & ranges.** ⚑ **T and λ are the assignment's named ablation knobs — tune them jointly** (they interact). T ∈ {1,2,4,8,16} (extend high if using long training / AdamW — 2026 temperature study shows τ≥10 can win with strong teachers); λ ∈ {0.1,0.3,0.5,0.7,0.9}. Reuse T3's LR/wd, small re-tune.
- **Agent/HPO strategy.** **Optuna owns the T×λ grid/TPE sweep** (this doubles as the required C9 ablation — see Task 11). Agent owns nothing new unless the KL/loss code needs repair. This is the cleanest place to produce the mandatory ablation heatmap (val-acc over T×λ).
- **Deliverables.** T×λ ablation heatmap; `resnet18_ternary_vanillaKD_best.pth`; T3-vs-T4 comparison table; per-teacher (LS 0 vs 0.1) comparison.
- **Ablations.** T sweep; λ sweep; teacher-LS; KD+CE vs KD-only.
- **Rubric.** KD formulation & integration (6); ablation (part of 6); pipeline stability (7).
- **Pitfalls.** Teacher grad leakage; forgetting `T²`; comparing T4 to T3 under different seeds/epochs (must be controlled).

---

### TASK 5 — Advanced knowledge distillation (beyond vanilla logits)

- **Role.** Investigate *what form* of knowledge transfers best to a ternary student. Each method needs a **hypothesis**, not just a loss added.
- **Central hypothesis.** Because the ternary student's function class is far smaller than the teacher's, **relation-preserving / attainable-target** knowledge transfers better than exact-value imitation. (See Section 4.)
- **Candidate methods (each with the problem it solves):**
  - **DKD (Decoupled KD, CVPR'22)** — splits target-class (TCKD) vs non-target-class (NCKD) KD with separate weights; *solves* the implicit suppression of non-target "dark knowledge" in vanilla KD. Cheap, logit-only, strong baseline.
  - **DIST (NeurIPS'22)** — replaces exact KL with **Pearson-correlation** matching of teacher/student relations; *solves* the "strong-teacher targets are unattainable, exact matching hurts" failure — directly relevant to a big-gap ternary student.
  - **Feature KD (FitNet/AT)** — match intermediate feature maps / attention; *solves* logit-only KD's blindness to representation. ⚑ Evidence (SQAKD) that *raw* feature KD can *hurt* at low bit — pair with the next item.
  - **QFD-style quantized-target feature KD (AAAI'23)** — match the **teacher's own features passed through the student's quantizer** (an "attainable target"); *solves* the FP32-feature-unreachable problem. This is our strongest feature-KD candidate.
  - **CRD (contrastive) / RKD / SP-KD (relational)** — transfer geometry/relations rather than absolute values; *solves* representation-space mismatch; RKD/SP can let a student exceed a teacher on structure.
  - **SemCKD / ReviewKD (cross-layer)** — learned attention over teacher layers / cross-stage review; *solves* the "which teacher layer feeds which student layer" mismatch (ResNet34 and ResNet18 have different depths).
- **Deep sub-tasks.** Implement 2–3 candidates behind flags (recommend **DKD**, **DIST**, and **QFD-style quantized-feature KD**); controlled comparison vs T4 vanilla KD, same teacher/split/budget/seeds.
- **Hyperparameters & ranges.** DKD α,β (target/non-target weights) ∈ [0.5,8]; DIST intra/inter weights; feature-loss weight γ log 1e-2…1; layer-pairing choice.
- **Agent/HPO strategy.** Agent owns the *loss implementations* (structural). Optuna owns each method's *numeric weights*. ⚑ Do **not** stack every loss — one hypothesis at a time (rubric rewards reasoning, penalizes kitchen-sink).
- **Deliverables.** One table: vanilla-KD vs DKD vs DIST vs QFD-feature, mean±std, vs T3/T4.
- **Ablations.** logits-only vs feature-only vs combined; raw-feature vs quantized-feature target (the QFD test); relation-loss on logits vs features.
- **Rubric.** KD formulation (6); pipeline (7); report critical discussion (6).
- **Pitfalls.** Feature-dimension mismatch (need 1×1 projection); relation losses need batch ≥128; ⚑ correlation losses noisy on small batches.

---

### TASK 6 — Quantization-aware knowledge transfer (the core research direction)

- **Role.** The project's intended research contribution: **transform the teacher's knowledge into a form the ternary student can actually represent** before transferring it.
- **Central hypothesis.** The bottleneck is not teacher knowledge but its **representation**: FP32-continuous vs ternary-discrete. Transfer should be `teacher → student-aware projection → quantization-aware target → ternary student`.
- **Candidate mechanisms.**
  - **Quantized-target distillation (QFD generalized)** — quantize teacher features/logits through the student's own scales before matching (attainable target).
  - **Quantization-error-aware KD** — weight the KD term per layer by that layer's measured `‖W−Ŵ‖` so heavily-damaged layers lean more on richer supervision.
  - **Layer-sensitivity weighting** — from the Task 13 probe, up-weight supervision at sensitive layers (conv1, FC) without relaxing ternary.
  - **SQAKD-style** — KL(FP↔quant) + discretization-error term, minimal hyperparameters, optionally label-free.
- **Proposed primary method (from our roadmap): QTRD — Quantized-Target Relational Distillation.** Match **relations** (DIST/RKD-style correlation) between teacher features **after the student's quantizer** and student features, per stage, + a correlation logit loss + CE. Combines QFD's attainable-target with DIST's relation-preservation at ternary precision. Novelty class: *plausibly novel combination* — closest prior work QFD (2023) + DIST (2022); **not** claimed as unqualified novel (see §10).
- **Hyperparameters & ranges.** per-stage feature weights; correlation vs MSE; quantized vs raw target (ablation); temperature for relational softmax.
- **Agent/HPO strategy.** Agent implements QTRD loss + quantizer coupling; Optuna tunes the weights. This is where the agentic loop earns its keep (structural code changes), but each accepted change must be explainable.
- **Deliverables.** QTRD vs best-T5 vs T4 vs T3 table; the "attainable-target" ablation (raw vs quantized teacher target) as the headline scientific result.
- **Rubric.** Pipeline & stability (7); KD formulation (6); report discussion (6).
- **Pitfalls.** Complexity creep — keep the mechanism *simple with a clear principle* (Section 4); guard training stability.

---

### TASK 7 — Optimization / STE research

- **Role.** Test whether the *gradient estimator*, not the loss, limits the ternary student.
- **Hypothesis.** Better-conditioned estimators (clipped/rectified) reduce STE bias and change whether KD helps at all — a quantizer-side fix independent of loss design.
- **Design space.** vanilla identity-STE (baseline/negative control) vs **clipped-STE** (Yin et al.-justified) vs **ReSTE** (ICCV'23, annealed power-function `sign(z)|z|^{1/o}`, o:1→3) vs gradient-scaling/clipping; learned α (TTQ) vs fixed; per-layer vs per-channel; adaptive thresholds.
- **Hyperparameters & ranges.** ReSTE `o` schedule; grad-clip norm; α LR (often decoupled from weight LR).
- **Agent/HPO.** Agent implements STE variants (structural); Optuna tunes their scalars.
- **Deliverables.** STE-variant comparison table on the fixed T3/T4 setup.
- **Rubric.** Ternary quantizer & architecture (6); pipeline (7).
- **Pitfalls.** Identity-STE can be unstable near minima (Yin et al.) — expected; document as the negative control.

---

### TASK 8 — Progressive / curriculum ternarization

- **Role.** Test whether imposing ternary immediately is too hard.
- **Hypothesis (⚑ falsifiable — do not assume it helps).** `FP32 → 8-bit → 4-bit → ternary` (or soft-to-hard threshold annealing) eases optimization and improves final accuracy. Literature has *little* controlled CNN evidence — design to falsify.
- **Design space.** bit-width curriculum vs threshold/temperature annealing vs QAT warmup (first k epochs FP, then quantize) vs delayed ternarization; KD-weight schedule coupled to quant stage.
- **Agent/HPO.** Agent implements the schedule; Optuna tunes stage lengths.
- **Deliverables.** progressive vs immediate comparison (same budget), loss-stability curves.
- **Rubric.** Pipeline & stability (7 — "evidence of debugging effort e.g. progressive quantization, warmup" is *named* in the rubric).
- **Pitfalls.** Longer schedules can just mean "more training" — control total epochs.

---

### TASK 9 — Adaptive knowledge transfer

- **Role.** Test whether per-sample KD treatment beats uniform KD.
- **Hypothesis.** Easy/high-confidence samples → stronger soft-target supervision; hard/ambiguous → rely more on hard labels or relational knowledge.
- **Design space.** sample-wise / confidence-based temperature; teacher-entropy or teacher-student-disagreement gating of λ; dynamic-temperature schedules (2025–2026 work). ⚑ Ties to carry-forward note A: an LS=0.1 teacher is over-smooth, which interacts with confidence signals — test with the LS=0 teacher too.
- **Agent/HPO.** Agent implements the adaptive rule; Optuna tunes its parameters.
- **Deliverables.** adaptive vs fixed-T/λ comparison.
- **Rubric.** KD formulation (6); report discussion (6).
- **Pitfalls.** Adaptive schemes add variance; validate before trusting.

---

### TASK 10 — Gradient-conflict-aware training

- **Role.** Test whether CE / KD / feature / quant gradients *conflict* and whether resolving conflict helps.
- **Hypothesis.** Useful teacher supervision is suppressed by competing hard-label/quant gradients in the reduced ternary manifold.
- **Deep sub-tasks.** Measure `cos(g_CE, g_KD)`, `cos(g_feat, g_relation)`, `cos(g_KD, g_quant)` during training (a diagnostic worth reporting regardless). Then try **PCGrad** (gradient surgery), **uncertainty weighting** (Kendall), **GoR-style learned loss balancing**, or gradient normalization.
- **Agent/HPO.** Agent implements the balancing/surgery; Optuna tunes any scalars.
- **Deliverables.** cosine-conflict plots + balanced-vs-static comparison.
- **Rubric.** Pipeline & stability (7); report discussion (6).
- **Pitfalls.** ⚑ Don't add complexity for its own sake — only if the cosine plots show real conflict.

---

### TASK 11 — Ablation studies (satisfies C9, aim broader)

- **Role.** Controlled ablations; **≥1 of T / λ / Δ is mandatory** (do this in T4 at minimum).
- **Matrix (change one factor at a time):** KD vs no-KD (T3 vs T4 — headline); T sweep; λ sweep; α strategy (tensor/layer/channel); threshold t; STE variant; feature vs no feature; relation vs no relation; progressive vs immediate; adaptive vs fixed; teacher LS 0 vs 0.1; augmentation.
- **Record for each:** accuracy (mean±std, 3 seeds), loss, gap, model size, sparsity, compression ratio.
- **Agent/HPO.** Optuna produces the numeric sweeps; export studies to the ablation tables/heatmaps.
- **Rubric.** Ablation (part of report's 6); Evaluation (5).
- **Pitfalls.** Uncontrolled confounds — same teacher/split/budget/seed policy throughout (experimental discipline).

---

### TASK 12 — Final compression analysis

- **Role.** The required comparison + honest compression accounting.
- **Deep sub-tasks.**
  1. Table: FP32 ResNet34 · FP32 ResNet18 · ternary ResNet18 → params, theoretical weight storage, sparsity (%zeros), accuracy, accuracy-drop, compression ratio.
  2. ⚑ **Compression math honesty:** ternary ≈ `log₂3 ≈ 1.58` bits/weight (plus per-channel α storage overhead), so theoretical ≈ ~20× vs FP32 — **but** state clearly that **real wall-clock speedup needs specialized ternary kernels/hardware**; PyTorch stores latent FP weights and runs dense FP kernels, so *measured* speed will not reflect the theoretical ratio. The rubric explicitly rewards this honesty.
  3. Compression-vs-accuracy scatter (teacher/FP-student/ternary-student).
  4. Run `verify_ternary.py` on the final checkpoint → confirm forward weights ∈ `{−α,0,+α}` per channel; distinguish latent vs deployed.
- **Rubric.** Evaluation & compression (5); limitations discussion (6).
- **Pitfalls.** Counting latent FP weights as inference weights; conflating theoretical bits with real speed.

---

## 4. Central Research Thesis (ties T5/T6 together)

> **The limitation is likely representational, not informational.** The teacher has the knowledge; the ternary student cannot represent the teacher's continuous FP32 feature geometry — forcing exact imitation supplies a loss gradient that can never reach zero and fights the CE objective (gradient conflict). The productive paradigm is **knowledge transformation into an attainable, relation-preserving target inside the ternary function class**, not raw imitation. This predicts: (a) relation/correlation KD (DIST/RKD) ≥ exact-value KD; (b) quantized-target feature KD (QFD/QTRD) > raw-feature KD; (c) vanilla KD may underperform no-KD (T4 vs T3). All three are directly testable in T4–T6 and are the report's scientific spine.

---

## 5. Master Hyperparameter / Search-Space Table

| Symbol | Meaning | Search range | Owner | Tune in |
|---|---|---|---|---|
| LR | learning rate (SGD) | log 1e-2…1e-1 (lower than FP32) | Optuna | T3–T6 |
| wd | weight decay (latent weights) | log 1e-5…5e-4 | Optuna | T3 |
| T | KD temperature | {1,2,4,8,16} | Optuna | **T4 (C9)** |
| λ | KD weight | {0.1,0.3,0.5,0.7,0.9} | Optuna | **T4 (C9)** |
| t | ternary threshold factor (Δ=t·max\|W\|) | 0.02…0.1 | Optuna | **T3 (C9)** |
| α | scale (per-tensor/layer/channel; sym/asym) | categorical | Optuna+agent | T3 |
| STE | estimator (identity/clipped/ReSTE) | categorical | agent | T7 |
| γ_feat | feature-loss weight | log 1e-2…1 | Optuna | T5/T6 |
| DKD α,β | target/non-target KD weights | 0.5…8 | Optuna | T5 |
| quant-target | raw vs student-quantized teacher feature | categorical | agent | T6 |
| schedule | immediate vs progressive/annealed | categorical | agent | T8 |

Sampler: `TPESampler(seed)` + `HyperbandPruner`, min-epochs ≈ 15–20 (QAT noise), SQLite study, 3-seed confirm on winners.

---

## 6. Controlled Experiment Ladder

`A` FP32 ResNet34 (done) → `B` FP32 ResNet18 no-KD (T2) → `C` ternary-18 no-KD (T3) → `D` ternary-18 + vanilla KD (T4) → `E` +DKD → `F` +best feature KD (T5) → `G` +relation KD → `H` +QFD/QTRD quantized-target (T6) → `I` best proposed method. Same teacher, split, budget, seed policy throughout. ⚑ **Row D vs C is the pivotal measurement.**

---

## 7. Compute Budget & Schedule (to 16 Sep)

~19 days, one GPU. ⚑ Our `AutoMLresearch.md` timeline overran to 22 Sep — corrected below:

| Days | Work |
|---|---|
| 28–30 Aug | Repo scaffold (§2.5); freeze `data.py`; **T2** FP32 ResNet18 (3 seeds); LS=0 teacher train |
| 31 Aug–3 Sep | **T3** ternary QAT baseline + quantizer/STE + `verify_ternary.py`; Optuna {LR,wd,t} |
| 4–7 Sep | **T4** vanilla KD; **T×λ ablation (C9)**; T3-vs-T4 + teacher-LS comparison |
| 8–11 Sep | **T5/T6** DKD + DIST + QFD/QTRD (the research contribution); pick winner |
| 12–13 Sep | Optional T7/T8/T10 if time; 3-seed confirm on the winner |
| 14–15 Sep | **T11/T12** ablation tables, compression analysis, all plots |
| 15–16 Sep | Report PDF, README end-to-end check, checkpoints, attach AI threads, submit |

⚑ Buffer built in before the 23:59 deadline. If compute slips, the *minimum viable pass* is A+B+C+D + T×λ ablation + T12 — that already covers C7/C8/C9 and most of the rubric; T5/T6 are the upside.

---

## 8. Deliverables Checklist (mapped to rubric)

- [ ] Code repo, runs end-to-end, documented README (all tasks)
- [ ] `data.py` frozen split; `ternary.py` (latent≠forward, α, Δ, STE) — **6 marks**
- [ ] KD losses (T²-scaled KL + CE; teacher frozen) + T,λ exploration — **6 marks**
- [ ] Stable KD+QAT pipeline, loss curves, warmup/progressive evidence — **7 marks**
- [ ] Eval table (teacher / FP32-18 / ternary-18), size+sparsity+compression, honest drop — **5 marks**
- [ ] Report: architecture diagrams, curves, ≥1 ablation (T/λ/Δ), limitations (kernels, early/late layers), citations — **6 marks**
- [ ] Checkpoints: best teacher + best ternary student + `verify_ternary.py`
- [ ] Reproducibility: seeds, requirements.txt/env, configs, history
- [ ] AI chat threads attached; honest autoresearch framing (§2.4, §10)

## 9. Risk Register

| Risk | Mitigation |
|---|---|
| Vanilla KD hurts ternary student | Report honestly (valid finding); pursue T5/T6 attainable-target methods |
| QAT training unstable/NaN | warm-start from T2, lower LR, grad-clip, progressive (T8), clipped-STE |
| Test-set leakage via repeated tuning | tune only on 5k val; test **once**; agent never sees test |
| autoresearch local-search stall / cost | cap trials, stage tuning, prefer Optuna for numbers, keep budget small |
| AI-disclosure penalty | attach threads; understand every line; honest framing; own the hypotheses |
| Compression over-claim | report theoretical bits AND the no-real-speedup-without-kernels caveat |

## 10. Honest AI-Usage & Novelty Statement (for the report)

- **AI usage:** planning and coding assistance (this thread), plus `autoresearch` (automated recipe/hyperparameter search within fixed architecture/dataset) and Optuna (numeric HPO). Framed as tuning/experimentation, **not** autonomous research. All AI chat threads attached; `autoresearch` git history included as the experiment log.
- **Novelty:** the QTRD/quantized-target-relational direction (T6) is positioned as *"to the best of our literature search, a not-previously-published combination"* — closest prior work QFD (AAAI'23) and DIST (NeurIPS'22); we make **no** unqualified novelty claim. TWN/TTQ/DKD/DIST/QFD/SQAKD/ReSTE and the required papers (He; Hinton; TWN; TTQ; Bengio STE; Yin et al.) are cited.

---

*Prepared as the standing plan. Next action: on "WORK ON TASK 2", scaffold the repo (§2.5), freeze the shared split, and train the FP32 ResNet18 baseline (and, in parallel, the LS=0 teacher control) — implementing Task 2 only.*
