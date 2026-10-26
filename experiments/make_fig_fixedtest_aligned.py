#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-test figures (Fig_fixedtest_rot_hit, Fig_fixedtest_official_panel, Fig_fixedtest_subset_fair) from the
time-aligned evaluation (fixedtest_aligned_results.json, inertial_classical_aligned_results.json) and the ORB-SLAM3
coverage of the monocular and the recalibrated mono-inertial runs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
IMG = HERE / "figures"
RES = Path(os.environ.get("FIXEDTEST_OUT", HERE / "fixedtest_aligned_results.json"))
from paths import IMUCAM_ROOT  # noqa: E402
MONO = IMUCAM_ROOT / "vio_orbslam3_mono_fixedtest_metrics.json"
MI = HERE / "orbslam3_mono_inertial_aligned_coverage.json"
C_IN, C_CL, C_VIS, C_FU = "#1f4e79", "#6b7280", "#9a3412", "#0f766e"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.tick_params(labelsize=8)


def bars(ax, labels, vals, colors, fmt, dy):
    b = ax.bar(labels, vals, color=colors, width=0.7)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + dy, fmt.format(v), ha="center", va="bottom", fontsize=7)
    ax.tick_params(axis="x", rotation=20)


def status_rows():
    mono = json.loads(MONO.read_text(encoding="utf-8"))
    per, ok = mono.get("per_session", {}), set(mono.get("tracking", {}).get("sessions_ok", []))
    mi = json.loads(MI.read_text(encoding="utf-8")) if MI.exists() else {}
    names = mi.get("sessions") or list(per)
    st_mono = [2 if isinstance(per.get(s), dict) and "error" not in per[s] and int(per[s].get("N", 0) or 0) > 0 else (1 if s in ok else 0)
               for s in names]
    st_mi = [mi["status"][s] for s in names] if mi else None
    return names, st_mono, st_mi


def main():
    R = json.loads(RES.read_text(encoding="utf-8"))
    rows, sub = R["rows"], R["subset"]
    cl = json.loads((HERE / "inertial_classical_aligned_results.json").read_text(encoding="utf-8"))["rows"]
    fw = json.loads((HERE / "fusion_weights_validation_results.json").read_text(encoding="utf-8"))["test_grid_weights"]
    lab = ["GRU6D", "Gyro", "Quat.", "ORB-VO", "ORB-VO\n+ GRU6D", "HF\n+ GRU6D"]
    rot = [rows["gru6d"]["rot_mean"], cl["gyro_zeroT"]["rot_mean"], cl["quat_zeroT"]["rot_mean"], rows["vo_only"]["rot_mean"],
           fw["vo"]["rot_mean"], fw["hf"]["rot_mean"]]
    hit = [rows["gru6d"]["hit_rot5"], cl["gyro_zeroT"]["hit_rot5"], cl["quat_zeroT"]["hit_rot5"], rows["vo_only"]["hit_rot5"],
           fw["vo"]["hit_rot5"], fw["hf"]["hit_rot5"]]
    col = [C_IN, C_CL, C_CL, C_VIS, C_FU, C_FU]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    bars(ax[0], lab, rot, col, "{:.2f}", 0.06)
    ax[0].set_ylabel("Rotation error mean (deg)")
    ax[0].set_title("(a) Full-coverage rotation error", fontsize=9)
    style(ax[0])
    bars(ax[1], lab, hit, col, "{:.3f}", 0.005)
    ax[1].set_ylim(0.85, 1.01)
    ax[1].set_ylabel("Hit@rot $\\leq 5^\\circ$")
    ax[1].set_title("(b) Full-coverage hit rate", fontsize=9)
    style(ax[1])
    for ext in ("pdf", "png"):
        fig.savefig(IMG / f"Fig_fixedtest_rot_hit.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    s = sub["on_slam_windows"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    bars(ax[0], ["ORB-SLAM3", "ORB-VO", "GRU6D", "Gyro"], [s["orbslam3"], s["orbvo"], s["gru6d"], s["gyro"]],
         ["#166534", C_VIS, C_IN, C_CL], "{:.2f}", 0.05)
    ax[0].set_ylabel("Rotation error mean (deg)")
    ax[0].set_title(f"(a) Same {s['N']} windows (ORB-SLAM3 tracked)", fontsize=9)
    style(ax[0])
    n = [s["N"], sub["on_orbvo_windows"]["N"], 2312, 2312]
    bars(ax[1], ["ORB-SLAM3", "ORB-VO", "GRU6D", "Gyro"], n, ["#166534", C_VIS, C_IN, C_CL], "{:d}", 20)
    ax[1].set_ylabel("Windows $N$ (all sessions)")
    ax[1].set_title("(b) Coverage", fontsize=9)
    style(ax[1])
    for ext in ("pdf", "png"):
        fig.savefig(IMG / f"Fig_fixedtest_subset_fair.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85])
    a = fig.add_subplot(gs[0, 0])
    bars(a, lab, rot, col, "{:.2f}", 0.06)
    a.set_ylabel("Rot. mean (deg)")
    a.set_title("(a) Full-coverage rotation error", fontsize=9)
    style(a)
    b = fig.add_subplot(gs[0, 1])
    pts = [("GRU6D", 2312, rows["gru6d"]["rot_mean"], C_IN), ("Gyro", 2312, cl["gyro_zeroT"]["rot_mean"], C_CL),
           ("ORB-VO", rows["vo_only"]["N"], rows["vo_only"]["rot_mean"], C_VIS), ("HF", rows["hf_only"]["N"], rows["hf_only"]["rot_mean"], "#b45309"),
           ("ORB-SLAM3", rows["slam_only"]["N"], rows["slam_only"]["rot_mean"], "#166534")]
    for name, nn, r, c in pts:
        b.scatter([nn], [r], s=60, c=c, zorder=3)
        dy = -11 if name == "Gyro" else 4
        b.annotate(name, (nn, r), textcoords="offset points", xytext=(-38 if nn > 2000 else 5, dy), fontsize=7, color=c)
    b.set_xlabel("Windows $N$")
    b.set_ylabel("Rot. mean (deg)")
    b.set_title("(b) Accuracy vs. coverage", fontsize=9)
    b.set_xlim(0, 2600)
    b.set_ylim(0, 6.0)
    style(b)
    c = fig.add_subplot(gs[1, 0])
    bars(c, ["ORB-SLAM3", "ORB-VO", "GRU6D", "Gyro"], [s["orbslam3"], s["orbvo"], s["gru6d"], s["gyro"]],
         ["#166534", C_VIS, C_IN, C_CL], "{:.2f}", 0.05)
    c.set_ylabel("Rot. mean (deg)")
    c.set_title(f"(c) {s['N']} windows tracked by ORB-SLAM3", fontsize=9)
    style(c)
    d = fig.add_subplot(gs[1, 1])
    names, st_mono, st_mi = status_rows()
    cmap = {0: "#b91c1c", 1: "#ca8a04", 2: "#166534"}
    for i, st in enumerate(st_mono):
        d.bar(i, 0.9, bottom=1.05 if st_mi else 0, color=cmap[st], width=0.85, edgecolor="white", linewidth=0.4)
    if st_mi:
        for i, st in enumerate(st_mi):
            d.bar(i, 0.9, color=cmap[st], width=0.85, edgecolor="white", linewidth=0.4)
        d.set_yticks([0.45, 1.5])
        d.set_yticklabels(["mono-inertial", "mono"], fontsize=7)
    else:
        d.set_yticks([])
    d.set_xticks(range(len(names)))
    d.set_xticklabels([f"S{i + 1}" for i in range(len(names))], fontsize=7)
    d.set_ylim(0, 2.45 if st_mi else 1.2)
    d.set_title("(d) ORB-SLAM3 session status", fontsize=9)
    for sp in ("top", "right", "left"):
        d.spines[sp].set_visible(False)
    d.legend(handles=[Patch(facecolor=cmap[2], label="RPE ok"), Patch(facecolor=cmap[1], label="traj. only"),
                      Patch(facecolor=cmap[0], label="fail")], loc="upper center", fontsize=7, frameon=False, ncol=3,
             bbox_to_anchor=(0.5, -0.12))
    for ext in ("pdf", "png"):
        fig.savefig(IMG / f"Fig_fixedtest_official_panel.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("wrote Fig_fixedtest_rot_hit, Fig_fixedtest_subset_fair, Fig_fixedtest_official_panel to", IMG)


if __name__ == "__main__":
    sys.exit(main())
