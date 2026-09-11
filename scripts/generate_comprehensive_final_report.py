#!/usr/bin/env python3
"""Generate the authoritative, reconciled ATDL final PDF report."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "ATDL_comprehensive_final_report.pdf"
OFFICIAL = ROOT / "results" / "final_official_test_evaluation.json"
STATE = ROOT / "research_state_checkpoint.json"


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def history_seconds(relative: str) -> float:
    payload = read(relative)
    return float(payload.get("elapsed_seconds", payload.get("elapsed_sec", 0.0)))


def fmt_pct(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{100 * value:.{digits}f}%"


def fmt_seconds(value: float) -> str:
    hours, remainder = divmod(int(round(value)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def report_image(path: Path, max_width: float = 7.0 * inch, max_height: float = 5.4 * inch) -> Image:
    from PIL import Image as PILImage
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def add_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4a5568"))
    canvas.drawString(0.65 * inch, 0.42 * inch, "ATDL - Knowledge Distillation with Ternary-Weight QAT")
    canvas.drawRightString(A4[0] - 0.65 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    if not OFFICIAL.is_file():
        raise FileNotFoundError("official final evaluation must exist before generating the report")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite final report: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    official = json.loads(OFFICIAL.read_text())
    task2 = read("results/task2_final_summary.json")
    task3 = read("results/task3_b2_default_summary.json")
    task4 = read("results/task4/task4_b3_final_t2_lam09_aggregate_summary.json")
    dkd = read("results/task4/task5_dkd_final_t2_lam09_a1_b8_r1_summary.json")
    dist = read("results/task4/task6_dist_screen_t2_lam09_i1_a1_r1_summary.json")
    dist_cont = read("results/task4/task7_dist_t1_lam05_r1_summary.json")
    qfd = read("results/task4/task8_qfd_screen_t2_lam09_aux1_r1_summary.json")
    at = read("results/task4/task9_at_screen_t2_lam09_aux1_r1_summary.json")
    rkd = read("results/task4/task10_rkd_screen_t2_lam09_aux1_r1_summary.json")
    qtrd = read("results/task4/task11_qtrd_finaltwo_screen_t2_lam09_aux1_r1_summary.json")

    teacher_seconds = history_seconds("training_history.json")
    task2_seconds = sum(float(row["elapsed_seconds"]) for row in task2["per_seed"])
    task3_seconds = sum(float(row["elapsed_seconds"]) for row in task3["per_seed"])
    task4_seconds = sum(history_seconds(row["history"]) for row in task4["per_seed"])
    research_seconds = sum(sum(float(row.get("elapsed_seconds", 0)) for row in item["per_seed"]) for item in (dkd, dist, dist_cont, qfd, at, rkd, qtrd))

    checkpoint_rows = [
        ("FP32 ResNet34 teacher", ROOT / "resnet34_cifar10_fp32_best.pth", 21_282_122, "FP32 deployed"),
        ("FP32 ResNet18 baseline", ROOT / "results/best_models/task2_b1_fp32_resnet18_best.pth", 11_173_962, "FP32 deployed"),
        ("Ternary ResNet18 no KD", ROOT / "results/best_models/task3_b2_default_best.pth", 11_173_962, "latent FP32 training checkpoint"),
        ("Vanilla ternary KD control", ROOT / "experiments/task4/checkpoints/task4_b3_final_t2_lam09_remaining_rerun1/resnet18_ternary_kd_seed44.pth", 11_173_962, "latent FP32 training checkpoint"),
        ("QTRD exploratory", ROOT / "results/task4/best_models/task11_qtrd_finaltwo_screen_t2_lam09_aux1_r1_best.pth", 11_173_962, "latent FP32 training checkpoint"),
    ]

    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, alignment=TA_CENTER, textColor=colors.HexColor("#17365d"), spaceAfter=14)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#4a5568"))
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#17365d"), spaceBefore=8, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=colors.HexColor("#1f4e79"), spaceBefore=7, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.2, leading=13, spaceAfter=6)
    small = ParagraphStyle("Small", parent=body, fontSize=7.7, leading=10)
    note = ParagraphStyle("Note", parent=body, fontSize=8.5, leading=11, leftIndent=10, textColor=colors.HexColor("#4a5568"))

    story = []
    story += [Paragraph("Advanced Topics in Deep Learning", title), Paragraph("Knowledge Distillation with Ternary-Weight Quantization-Aware Training", subtitle), Spacer(1, 0.18 * inch)]
    story += [Paragraph("Comprehensive Final Report", ParagraphStyle("Cover", parent=title, fontSize=20, leading=25)), Spacer(1, 0.12 * inch)]
    story += [paragraph("<b>Teacher:</b> CIFAR-adapted ResNet34 FP32<br/><b>Student:</b> CIFAR-adapted ResNet18 with strict ternary Conv/FC deployment<br/><b>Dataset:</b> CIFAR-10, fixed 45,000/5,000 train/validation split plus one final 10,000-image official test evaluation<br/><b>Final report generation:</b> " + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"), subtitle)]
    story += [Spacer(1, 0.35 * inch), paragraph("<b>Executive result.</b> The frozen vanilla ternary-KD control achieved " + fmt_pct(official["groups"]["Vanilla ternary KD (final control)"]["test_accuracy_mean"]) + " +/- " + fmt_pct(official["groups"]["Vanilla ternary KD (final control)"]["test_accuracy_std"]) + " on the official unseen CIFAR-10 test set across three independently trained, validation-selected checkpoints. This exceeds the ternary no-KD baseline by 0.18 percentage points on test accuracy while retaining strict ternary Conv/FC deployment.", body)]
    story += [Paragraph("Integrity statement", h1), paragraph("All architecture and hyperparameter choices were made using the fixed validation split. The final official test report was generated only after the teacher, baselines, final vanilla-KD control, and exploratory QTRD checkpoint were frozen. No test-based model selection, tuning, restart, or overwrite was permitted. The official evaluator records independently recomputed validation accuracy, test accuracy, class confusion matrices, and ternary-verification diagnostics for every tested checkpoint.", body), PageBreak()]

    story += [Paragraph("1. Problem, protocol, and reproducibility", h1)]
    story += [paragraph("The project trains a CIFAR-adapted ResNet34 teacher from scratch and transfers knowledge to ResNet18 students. The production data pipeline fixes a class-balanced 45,000/5,000 split from CIFAR-10 training examples (seed 42), with exact train and validation index hashes stored by the pipeline. Training uses random crop plus horizontal flip; validation and official test use deterministic normalization. CUDA execution, package locks, per-run configuration, seeds, checkpoints, histories, and reload checks are preserved.", body)]
    story += [paragraph("The official test set was inaccessible during all development. The one-time final evaluator was explicitly unlocked after the validation-selected checkpoints were frozen. It produces <i>results/final_official_test_evaluation.json</i> immutably; a second invocation refuses to overwrite the report.", note)]
    story += [Paragraph("2. Model and ternary-QAT design", h1)]
    story += [paragraph("The teacher is a CIFAR ResNet34 with a 3x3, stride-1 stem, no ImageNet max-pool, and a 10-class classifier. The student is the matching CIFAR ResNet18. In the compliant student, every convolution and the final fully connected layer deploys ternary weights. BatchNorm and biases remain FP32. For latent weight W, the per-output-channel threshold is Delta = 0.7 mean(|W|); the code Q(W) is in {-1, 0, +1}; and the deployed weight is W_hat = alpha Q(W), where alpha is the mean active absolute latent weight. Latent FP32 weights are optimized, while W_hat is used in every forward pass.", body)]
    story += [paragraph("Backpropagation uses a clipped straight-through estimator. This is genuine QAT rather than post-hoc ternarization: quantization is active in the forward pass throughout optimization. The saved ternary checkpoints are rebuilt and verified layer by layer: deployed values match the quantizer, every Conv/FC is covered, and latent parameters contain more than three values.", body)]
    story += [Paragraph("3. KD formulation and training", h1)]
    story += [paragraph("The final control uses vanilla KD: (1-lambda) CE(y, s) + lambda T^2 KL(softmax(t/T) || softmax(s/T)), with T=2 and lambda=0.9. The ResNet34 teacher is loaded from its frozen checkpoint, switched to eval mode, excluded from the optimizer, and evaluated under no-grad. Student training uses SGD with momentum 0.9, Nesterov, warmup plus cosine learning rate schedule, label smoothing 0.1, and the fixed validation protocol. The final control was trained for 200 epochs with seeds 42, 43, and 44.", body), PageBreak()]

    story += [Paragraph("4. Methods explored and decisions", h1)]
    methods = [
        ["Method", "Evidence", "Validation result", "Decision"],
        ["FP32 ResNet34 teacher", "200 epochs, one frozen teacher", "95.54%", "reference"],
        ["FP32 ResNet18 baseline", "200 epochs, 3 seeds", "95.34% +/- 0.14%", "baseline"],
        ["Ternary ResNet18, no KD", "200 epochs, 3 seeds", "95.21% +/- 0.08%", "quantization reference"],
        ["Vanilla ternary KD", "T=2, lambda=0.9, 200 epochs, 3 seeds", "95.12% +/- 0.15%", "frozen final KD control"],
        ["DKD", "3 seeds", "94.88% +/- 0.16%", "rejected below control"],
        ["DIST", "one-seed screen", "94.94%", "screening only"],
        ["DIST continuation", "T=1, lambda=0.5, one seed", "95.02%", "provisional below control"],
        ["QFD / attention transfer / RKD", "one seed each", "95.00% / 95.02% / 95.12%", "provisional screens"],
        ["QTRD", "one seed, 50 epochs", "95.06%", "exploratory; below control"],
    ]
    table = Table([[paragraph(cell, small) for cell in row] for row in methods], colWidths=[1.45*inch, 2.18*inch, 1.32*inch, 1.55*inch], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365d")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#b7c9d6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#edf3f7")]), ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
    story += [table, Spacer(1, .12*inch)]
    story += [paragraph("Automated research infrastructure was used to preserve staged configurations, append-only artifacts, diagnostics, and a research ledger. Open-ended AutoML/HPO was intentionally paused rather than allowed to conduct opaque or test-driven search. The published final selection is the pre-test, three-seed vanilla-KD aggregate; advanced one-seed methods are clearly labelled exploratory and are not promoted to final claims.", body)]
    story += [Paragraph("5. Final official unseen-test evaluation", h1)]
    test_rows = [["Model", "Validation mean", "Official test mean", "n", "Interpretation"]]
    val_map = {"FP32 ResNet34 teacher": .9554, "FP32 ResNet18 baseline": task2["validation_mean"], "Ternary ResNet18 (no KD)": task3["validation_mean"], "Vanilla ternary KD (final control)": task4["validation_mean"], "QTRD exploratory screen": qtrd["validation_mean"]}
    interpretations = {"FP32 ResNet34 teacher": "teacher reference", "FP32 ResNet18 baseline": "full-precision baseline", "Ternary ResNet18 (no KD)": "quantization cost reference", "Vanilla ternary KD (final control)": "selected before test access", "QTRD exploratory screen": "screen only; not selected"}
    for name, group in official["groups"].items():
        test_rows.append([name, fmt_pct(val_map[name]), fmt_pct(group["test_accuracy_mean"]) + " +/- " + fmt_pct(group["test_accuracy_std"]), str(group["n_checkpoints"]), interpretations[name]])
    test_table = Table([[paragraph(cell, small) for cell in row] for row in test_rows], colWidths=[1.72*inch, 1.13*inch, 1.55*inch, .35*inch, 2.0*inch], repeatRows=1)
    test_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365d")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#b7c9d6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#edf3f7")]), ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
    story += [test_table, Spacer(1, .12*inch), paragraph("The final vanilla-KD control improves official test accuracy over the no-KD ternary baseline by 0.18 percentage points (94.87% versus 94.69%). It remains 0.14 percentage points below the FP32 ResNet18 baseline and 0.32 percentage points below the ResNet34 teacher, quantifying the accuracy/compression trade-off honestly.", body), PageBreak()]

    story += [Paragraph("6. Training curves and KD stability", h1), report_image(ROOT / "plots/final_official/teacher_student_pre_post_kd_curves.png"), Spacer(1, .08*inch), paragraph("The trajectories compare the teacher, FP32 ResNet18, pre-KD ternary QAT, post-KD vanilla ternary QAT, and QTRD screen using real preserved histories. The shorter QTRD curve is deliberately not visually or statistically equated with the 200-epoch, three-seed control.", note), report_image(ROOT / "plots/final_official/vanilla_kd_stability_diagnostics.png"), PageBreak()]
    story += [Paragraph("7. Accuracy, ablation, compression, and error analysis", h1), report_image(ROOT / "plots/final_official/validation_vs_official_test.png"), Spacer(1, .08*inch), report_image(ROOT / "plots/final_official/kd_temperature_lambda_ablation.png", max_height=4.5*inch), PageBreak()]
    story += [Paragraph("8. Compression and classwise behavior", h1), report_image(ROOT / "plots/final_official/compression_and_sparsity.png", max_height=4.4*inch), Spacer(1, .08*inch), report_image(ROOT / "plots/final_official/official_test_confusion_teacher_vs_kd.png", max_height=4.3*inch), PageBreak()]

    story += [Paragraph("9. Checkpoints, storage, and execution time", h1)]
    check_rows = [["Artifact", "Parameters", "On-disk checkpoint", "Interpretation"]]
    for name, path, params, meaning in checkpoint_rows:
        check_rows.append([name, f"{params:,}", f"{mib(path):.2f} MiB", meaning])
    check_table = Table([[paragraph(cell, small) for cell in row] for row in check_rows], colWidths=[2.05*inch, 1.25*inch, 1.4*inch, 2.05*inch], repeatRows=1)
    check_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365d")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#b7c9d6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#edf3f7")]), ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
    story += [check_table, Spacer(1,.1*inch)]
    ideal_ratio = 32 / math.log2(3)
    story += [paragraph(f"The approximately 85 MiB ternary checkpoint files intentionally retain latent FP32 training weights, optimizer-independent state, and metadata so they can be resumed and verified. They are not deployment packages. The deployed Conv/FC representation requires log2(3) bits per weight plus scale metadata, giving an ideal weight-storage ratio of {ideal_ratio:.2f}x versus FP32 before packing, scale overhead, and runtime/kernel effects. The final-control ternary verification reports approximately 47% zero values; no hardware speedup is claimed.", body)]
    time_rows = [["Phase", "Recorded cumulative GPU training time"], ["Teacher (one 200-epoch run)", fmt_seconds(teacher_seconds)], ["FP32 ResNet18 baseline (3 seeds)", fmt_seconds(task2_seconds)], ["Ternary no-KD QAT baseline (3 seeds)", fmt_seconds(task3_seconds)], ["Vanilla ternary KD final control (3 seeds)", fmt_seconds(task4_seconds)], ["Advanced KD/QAT screens and confirmations", fmt_seconds(research_seconds)], ["Recorded cumulative total", fmt_seconds(teacher_seconds + task2_seconds + task3_seconds + task4_seconds + research_seconds)]]
    time_table = Table([[paragraph(cell, small) for cell in row] for row in time_rows], colWidths=[3.55*inch, 3.2*inch])
    time_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365d")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#b7c9d6")), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#edf3f7")]), ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
    story += [time_table, Spacer(1,.08*inch), paragraph("Times are sums of per-run saved elapsed times and represent active training duration; they exclude dataset preparation, notebook startup, report generation, and final inference-only evaluation. This makes the stated total auditable rather than a speculative wall-clock estimate.", note), PageBreak()]

    story += [Paragraph("10. Limitations and critical discussion", h1)]
    story += [paragraph("The ternary result is a storage-oriented algorithmic compression result. Practical inference acceleration requires ternary-aware kernels, efficient packing, memory-transfer analysis, and deployment benchmarking; these were not claimed or measured. The model uses a fixed split and multiple seed checks, but exact deterministic equivalence is not claimed for every CUDA operation. Advanced transfer losses beyond vanilla KD were intentionally constrained to one-seed screens unless evidence justified further expenditure; they should not be interpreted as confirmatory comparisons. Representation-similarity metrics were not logged in the completed early runs and are therefore not fabricated.", body)]
    story += [Paragraph("11. Required references", h1)]
    refs = [
        "K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.",
        "G. Hinton, O. Vinyals, and J. Dean. Distilling the Knowledge in a Neural Network. NeurIPS Deep Learning Workshop, 2015.",
        "F. Li, B. Zhang, and B. Liu. Ternary Weight Networks. arXiv:1605.04711, 2016.",
        "C. Zhu, S. Han, H. Mao, and W. J. Dally. Trained Ternary Quantization. ICLR, 2017.",
        "Y. Bengio, N. Leonard, and A. Courville. Estimating or Propagating Gradients Through Stochastic Neurons. arXiv:1308.3432, 2013.",
        "P. Yin, S. Lyu, R. Zhang, S. Osher, Y. Qi, and J. Xin. Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets. ICLR, 2019.",
    ]
    for idx, ref in enumerate(refs, 1): story.append(paragraph(f"[{idx}] {ref}", body))
    story += [Paragraph("Authoritative artifacts", h1), paragraph("Official results: results/final_official_test_evaluation.json. Visual notebook: ATDL_final_official_evaluation_visual_report.ipynb. Final selected checkpoint: experiments/task4/checkpoints/task4_b3_final_t2_lam09_remaining_rerun1/resnet18_ternary_kd_seed44.pth. Ternary verification tool: src/evaluation/verify_ternary.py. This report supersedes the earlier validation-only final report for submission-facing claims.", body)]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=.6*inch, leftMargin=.6*inch, topMargin=.62*inch, bottomMargin=.62*inch, title="ATDL Comprehensive Final Report", author="ATDL Project")
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)

    state = json.loads(STATE.read_text())
    state.update({
        "timestamp": datetime.now().astimezone().isoformat(),
        "test_evaluation": "COMPLETED_FINAL_REPORTING_ONLY",
        "official_final_evaluation": "results/final_official_test_evaluation.json",
        "authoritative_final_report": "output/pdf/ATDL_comprehensive_final_report.pdf",
        "official_test_selection_policy": official["selection_policy"],
        "final_method": "vanilla KD (Task 4 control; frozen before official test evaluation)",
    })
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
