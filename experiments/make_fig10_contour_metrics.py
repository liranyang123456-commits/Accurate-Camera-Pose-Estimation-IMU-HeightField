#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the updated Figure 10 (Fig_contour_metrics_panel.pdf / .png):
Quantitative contour and structural support performance across CholecSeg8k (N=8080),
CholecSeg8k-Hard stratified subsets, BSDS500 (N=500), and BIPEDv2 (N=250).
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
})


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.6), dpi=300)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # -------------------------------------------------------------------------
    # (a) CholecSeg8k Soft-Tissue Paired Region Overlap (N = 8,080 frames)
    # -------------------------------------------------------------------------
    methods_a = [
        "Ours (q=0.68 fill)",
        "MedSAM (ViT-B)*",
        "FastSAM-x",
        "YOLO11m-seg",
        "Ours (q=0.68 raw)",
        "YOLO11s-seg",
        "SAM 2.1-B",
        "SAM 2.1-L",
    ]
    dice_a = [0.769, 0.736, 0.602, 0.589, 0.561, 0.557, 0.499, 0.257]
    iou_a = [0.641, 0.586, 0.460, 0.540, 0.401, 0.494, 0.467, 0.190]

    x_a = np.arange(len(methods_a))
    w_a = 0.36
    b1 = ax_a.bar(x_a - w_a / 2, dice_a, w_a, label="Dice ↑", color="#1d4ed8", edgecolor="black", linewidth=0.4)
    b2 = ax_a.bar(x_a + w_a / 2, iou_a, w_a, label="IoU ↑", color="#60a5fa", edgecolor="black", linewidth=0.4)
    b1[0].set_facecolor("#0f766e")
    b2[0].set_facecolor("#2dd4bf")
    for rect in (b1[0], b2[0]):
        rect.set_edgecolor("#042f2e")
        rect.set_linewidth(0.9)
    for idx_bar, val in enumerate(dice_a):
        ax_a.text(x_a[idx_bar] - w_a / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=6.6,
                  fontweight="bold" if idx_bar == 0 else "normal")
    ax_a.axhline(0.813, color="black", linestyle="--", linewidth=1.0, label="All-foreground Dice (0.813)")
    ax_a.axhline(0.690, color="#6b7280", linestyle="--", linewidth=1.0, label="All-foreground IoU (0.690)")
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(methods_a, rotation=24, ha="right")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel("Overlap Score")
    ax_a.set_title("(a) CholecSeg8k Soft-Tissue Region Overlap (N = 8,080)", fontweight="bold")
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    ax_a.legend(loc="upper right", ncol=2, frameon=True, fontsize=6.8)

    # -------------------------------------------------------------------------
    # (b) CholecSeg8k-Hard Stratified Nuisance Subsets (Ours q=0.68 filled)
    # -------------------------------------------------------------------------
    subsets_b = [
        "All\n(8,080)",
        "Clean\n(6,024)",
        "Hard Union\n(2,056)",
        "Specular\n(808)",
        "Blur\n(808)",
        "Blood\n(625)",
    ]
    dice_b = [0.769, 0.767, 0.775, 0.790, 0.784, 0.723]
    iou_b = [0.641, 0.638, 0.649, 0.660, 0.673, 0.575]
    bf1_b = [0.269, 0.273, 0.257, 0.207, 0.322, 0.235]

    x_b = np.arange(len(subsets_b))
    w_b = 0.25
    ax_b.bar(x_b - w_b, dice_b, w_b, label="Dice ↑", color="#0f766e", edgecolor="black", linewidth=0.4)
    ax_b.bar(x_b, iou_b, w_b, label="IoU ↑", color="#14b8a6", edgecolor="black", linewidth=0.4)
    ax_b.bar(x_b + w_b, bf1_b, w_b, label="Boundary F1 ↑", color="#f59e0b", edgecolor="black", linewidth=0.4)
    allfg_b = [0.813, 0.803, 0.842, 0.814, 0.853, 0.868]
    ax_b.scatter(x_b - w_b, allfg_b, marker="_", s=260, linewidths=2.0, color="black", zorder=5, label="All-foreground Dice")
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(subsets_b)
    ax_b.set_ylim(0, 1.08)
    ax_b.set_ylabel("Score")
    ax_b.set_title("(b) Stratified Nuisance Resilience on CholecSeg8k-Hard", fontweight="bold")
    ax_b.grid(axis="y", linestyle="--", alpha=0.35)
    ax_b.legend(loc="upper right", ncol=2, frameon=True)

    # -------------------------------------------------------------------------
    # (c) Boundary Precision & F1 on BSDS500 Test (N=200) and BIPEDv2 Test (N=50)
    # -------------------------------------------------------------------------
    methods_c = [
        "Ours (q=0.68)",
        "Ours (q=0.90)",
        "Ours (q=0.95)",
        "Canny",
        "Sobel",
        "Laplacian",
        "DoG",
        "YOLO11m-seg",
        "SAM 2.1-L",
        "FastSAM-x",
    ]
    bsds_f1 = [0.370, 0.315, 0.301, 0.358, 0.409, 0.394, 0.393, 0.221, 0.213, 0.144]
    biped_prec = [0.647, 0.699, 0.709, 0.556, 0.450, 0.320, 0.307, 0.519, 0.255, 0.140]
    biped_f1 = [0.421, 0.319, 0.296, 0.492, 0.537, 0.430, 0.417, 0.035, 0.093, 0.048]

    x_c = np.arange(len(methods_c))
    w_c = 0.26
    ax_c.bar(x_c - w_c, biped_prec, w_c, label="BIPEDv2 B-Prec ↑", color="#1e40af", edgecolor="black", linewidth=0.4)
    ax_c.bar(x_c, biped_f1, w_c, label="BIPEDv2 B-F1 ↑", color="#3b82f6", edgecolor="black", linewidth=0.4)
    ax_c.bar(x_c + w_c, bsds_f1, w_c, label="BSDS500 B-F1 ↑", color="#93c5fd", edgecolor="black", linewidth=0.4)
    ax_c.text(x_c[2] - w_c, biped_prec[2] + 0.015, "0.709", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax_c.text(x_c[0], biped_f1[0] + 0.015, "0.421", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(methods_c, rotation=24, ha="right")
    ax_c.set_ylim(0, 0.84)
    ax_c.set_ylabel("Boundary Metric (t = 1 px)")
    ax_c.set_title("(c) Cross-Domain Boundary Precision & F1 (BIPEDv2 & BSDS500)", fontweight="bold")
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper right", frameon=True)

    # -------------------------------------------------------------------------
    # (d) Contour Geometric Distance on CholecSeg8k (N=8,080): Chamfer & HD95 (px)
    # -------------------------------------------------------------------------
    methods_d = [
        "FastSAM-x",
        "DoG",
        "Sobel",
        "Ours (q=0.68)",
        "Ours (q=0.95)",
        "SAM 2.1-L",
        "YOLO11m-seg",
        "YOLO11s-seg",
        "SAM 2.1-B",
    ]
    chamfer_d = [21.2, 33.9, 35.1, 43.5, 91.8, 114.7, 198.5, 290.3, 337.7]
    hd95_d = [138.8, 148.4, 147.9, 162.3, 313.0, 277.7, 320.1, 413.0, 477.4]

    x_d = np.arange(len(methods_d))
    w_d = 0.36
    b_ch = ax_d.bar(x_d - w_d / 2, chamfer_d, w_d, label="Chamfer Dist. (px) ↓", color="#be123c", edgecolor="black", linewidth=0.4)
    b_hd = ax_d.bar(x_d + w_d / 2, hd95_d, w_d, label="HD95 (px) ↓", color="#fb7185", edgecolor="black", linewidth=0.4)
    b_ch[3].set_facecolor("#0f766e")
    b_hd[3].set_facecolor("#2dd4bf")
    ax_d.text(x_d[3] - w_d / 2, chamfer_d[3] + 8, "43.5", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax_d.text(x_d[3] + w_d / 2, hd95_d[3] + 8, "162.3", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(methods_d, rotation=24, ha="right")
    ax_d.set_ylim(0, 540)
    ax_d.set_ylabel("Boundary Distance Error (px) ↓")
    ax_d.set_title("(d) Contour Geometric Error (Chamfer & HD95 on CholecSeg8k)", fontweight="bold")
    ax_d.grid(axis="y", linestyle="--", alpha=0.35)
    ax_d.legend(loc="upper left", frameon=True)

    fig.tight_layout(pad=1.1)
    out_dir = Path(__file__).resolve().parent / "figures"
    out_pdf = out_dir / "Fig_contour_metrics_panel.pdf"
    out_png = out_dir / "Fig_contour_metrics_panel.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
