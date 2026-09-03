# Task 3 — Ternary ResNet18 QAT Baseline (B2, no KD)

Drop-in bundle implementing **Task 3 only**: a ResNet18 whose **all** conv + FC
weights are ternary `{-α, 0, +α}`, trained with the ternary constraint **active
during training** (genuine QAT — not post-hoc PTQ), with **no KD**. This is the
pure-quantization reference (B2) that isolates the cost of ternarization vs the
Task 2 FP32 student.

Unzip at your repo root (it adds to the Task 2 tree; it depends on `src/data.py`,
`src/models/resnet_cifar.py`, `src/training/utils.py` from Task 2 — do not change those).

## New files

```
src/quant/
  ├── ste.py                       # STE variants: clipped (default, Yin et al.) + identity (Bengio)
  └── ternary.py                   # TernaryConv2d/Linear, TWN+TTQ quantizers, convert_to_ternary, warm-start
src/evaluation/verify_ternary.py   # ternary-constraint verifier (C10 / Phase 37)
scripts/train_student_ternary.py   # Task 3 QAT trainer (no KD); also hosts Tasks 4-6 later
configs/ternary/resnet18_ternary_noKD.yaml
```

## Quantizer (rubric: "Ternary quantizer & student architecture", 6 marks)

- **Latent vs deployed (C15):** each layer keeps a full-precision latent `weight`
  (trainable); the forward pass always uses `Ŵ = α · Q(W)`, `Q(W) ∈ {-1,0,+1}`.
  Latent weights are never used at inference and never reported as deployed.
- **Threshold Δ:** `twn` = 0.7·mean|W| (default) or `ttq` = t·max|W| (t=0.05).
- **Scale α:** `derived` = TWN-optimal `mean_{|W|>Δ}|W|` (default) or `learned` (TTQ param).
- **Symmetry:** `{-α,0,+α}` (default, matches the assignment set) or `{-α_n,0,+α_p}` (TTQ, needs learned scales).
- **Scope:** `per_channel` (default) or `per_tensor`.
- **STE:** `clipped` (default; `dL/dW = dL/dQ·1[|W|≤1]`, Yin et al. 2019) or `identity` (Bengio et al.).
- **Default baseline = per-channel symmetric TWN (derived α) + clipped STE** — robust, exactly `{-α,0,+α}`, well-cited.
- **All conv + FC ternary, including first conv and FC (C14).** `--keep-first-last-fp32` exists ONLY as a diagnostic for the layer-sensitivity probe and must stay off for the compliant baseline.

## Run

```bash
# quick pipeline check first (1 short seed)
uv run scripts/train_student_ternary.py --seeds 42 --epochs 5 --smoke

# Task 3 baseline: 3 seeds, warm-started from the Task 2 FP32 student
uv run scripts/train_student_ternary.py

# verify the ternary constraint on the saved best model (deliverable)
uv run src/evaluation/verify_ternary.py \
    --checkpoint results/best_models/resnet18_ternary_noKD_best.pth

# --- Task-3 ablation knobs (serve the required ablation set) ---
# TTQ (learned, asymmetric, per-channel):
uv run scripts/train_student_ternary.py --seeds 42 --threshold ttq --scale learned --asymmetric
# per-layer/tensor scale, identity STE, no warm-start, etc. are all flags.
```

## What it produces
- `experiments/checkpoints/resnet18_ternary_noKD_seed{S}.pth` (each stores its `quant_config`)
- `results/best_models/resnet18_ternary_noKD_best.pth`
- `experiments/histories/…json`, `plots/…{loss,acc,gap,lr}.png`
- `results/resnet18_ternary_noKD_summary.json` (val mean±std + sparsity + all-ternary flag)

## Constraints honored
- **C4/C14** all conv + FC weights ternary (verified per channel; first conv + FC included).
- **C5** ternary active every step during training (QAT, not PTQ).
- **C9/C11** test set untouched; selection on the 5k val split; test only via `evaluate.py --final-evaluation`.
- **C15** latent FP32 ≠ deployed ternary (asserted by `verify_ternary`).
- **Scope** Task 3 only: `--kd-mode` must be `none` (raises otherwise; KD is Task 4).

## Verified before delivery (cloud, synthetic data — CIFAR download is proxy-blocked here but stock)
- 20 ternary conv + 1 ternary FC; forward exactly ternary per channel (all configs: TWN/TTQ, sym/asym, per-channel/tensor, clipped/identity) ✓
- latent≠deployed (continuous latent, ≤3 values/channel deployed) ✓
- STE gradient flows to latent weights; clipped STE masks `|W|>1` ✓
- warm-start copies FP32 weights into latent ✓
- 3 QAT steps stable (finite loss, weights update, still ternary) ✓
- `verify_ternary`: sparsity ≈ 42%, all_ternary=True, ~20.2× theoretical compression; checkpoint→rebuild→verify round-trips ✓
- weight decay applied to latent conv/linear weights only (BN/bias/α excluded) ✓
- `--kd-mode` guard raises for anything but `none` ✓

## Expected outcome (not a claim)
A warm-started per-channel-TWN ternary ResNet18 typically recovers most of the FP32
ResNet18 accuracy on CIFAR-10 (TTQ-style results report small drops). The
**Task 2 (FP32) − Task 3 (ternary)** gap is the **quantization cost**; combined with the
teacher−FP32-student gap, this decomposes the full accuracy budget that Task 4+ (KD)
then tries to recover. Report per-seed mean±std and the sparsity/α histograms from
`verify_ternary`.

## Next task
On **"WORK ON TASK 4"**: add vanilla KD (`(1−λ)·CE + λ·T²·KL`) to this exact QAT
pipeline using the frozen teacher(s), and run the mandatory **T×λ ablation** plus the
pivotal **T3 (no-KD) vs T4 (KD)** comparison — reusing this quantizer and the firewall.
