#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures and table rows for the visual benchmarks from the re-evaluated results (reeval_benchmarks.py); writes to experiments/figures/."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
IMG = HERE / "figures"
from paths import ENDOSIM_ROOT  # noqa: E402
SOTA = ENDOSIM_ROOT / "results" / "sota"
sys.path.insert(0, str(ENDOSIM_ROOT))
from endosim.eval.align import align_trajectories  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.6, "axes.labelsize": 8,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 7, "legend.fontsize": 6.6})
LAB = {"heightfield": "Ours (height field)", "static": "Static (reference)", "vggt": "VGGT", "endodac": "EndoDAC", "cut3r": "CUT3R",
       "reloc3r": "Reloc3r", "afsfm": "AF-SfMLearner", "endofast3r": "Endo-FASt3R", "pi3": "Pi3", "droid": "DROID-SLAM",
       "dust3r": "DUSt3R", "endo3r": "Endo3R", "mast3r": "MASt3R"}
CITE = {"vggt": "wang2025vggt", "endodac": "cui2024endodac", "cut3r": "wang2025cut3r", "reloc3r": "dong2025reloc3r", "afsfm": "shao2022afsfm",
        "endofast3r": "zeinoddin2025endofast3r", "pi3": "wang2025pi3", "droid": "teed2021droid", "dust3r": "leroy2024dust3r",
        "endo3r": "guo2025endo3r", "mast3r": "leroy2024mast3r"}


def load(name):
    return json.loads((HERE / "hf_benchmarks" / f"{name}_reeval.json").read_text(encoding="utf-8"))


def bars(ax, items, title, ylabel, median=True):
    for i, (k, mean, med) in enumerate(items):
        col = "#0f766e" if k == "heightfield" else ("#9ca3af" if k == "static" else "#3b82f6")
        hatch = "//" if k == "static" else None
        ax.bar(i - (0.2 if median else 0), mean, 0.38 if median else 0.6, color=col, edgecolor="black", linewidth=0.4, hatch=hatch)
        if median:
            ax.bar(i + 0.2, med, 0.38, color=col, alpha=0.5, edgecolor="black", linewidth=0.4, hatch=hatch)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([LAB[k] + ("$^\\dagger$" if k in ("dust3r", "endo3r", "mast3r") else "") for k, _, _ in items], rotation=35, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def fig10():
    r18, s92, c3 = load("real18")["summary"], load("sim92")["summary"], load("c3vd10")["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4))
    it = sorted([(k, v["sim3_mean"], v["sim3_median"]) for k, v in r18.items() if "sim3_mean" in v], key=lambda x: -x[1])
    bars(axes[0, 0], it, "(a) 18 real 4K-referenced segments", "Sim(3) ATE (mm)")
    it = sorted([(k, v["translating"]["sim3_mean"], v["translating"]["sim3_median"]) for k, v in s92.items() if "translating" in v], key=lambda x: -x[1])
    bars(axes[0, 1], it, "(b) 71 simulated sequences with camera translation", "Sim(3) ATE (mm)")
    it = sorted([(k, v["orbital"]["rpe_r_mean"], v["orbital"]["rpe_r_median"]) for k, v in s92.items() if "orbital" in v], key=lambda x: -x[1])
    bars(axes[1, 0], it, "(c) 21 simulated pure-rotation sequences", "RPE$_1$ rotation (deg)")
    it = sorted([(k, v["sim3_mean"], v["sim3_median"]) for k, v in c3.items() if "sim3_mean" in v], key=lambda x: -x[1])
    bars(axes[1, 1], it, "(d) 10 C3VD phantom sequences", "Sim(3) ATE (mm)")
    axes[0, 1].legend(handles=[Patch(facecolor="#3b82f6", edgecolor="black", label="Learned / SLAM baselines"),
                               Patch(facecolor="#0f766e", edgecolor="black", label="Ours (training-free)"),
                               Patch(facecolor="#9ca3af", edgecolor="black", hatch="//", label="Static reference")],
                      loc="upper right", fontsize=6.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(IMG / f"Fig_extended_vi_benchmark.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def fmt_p(p):
    if p >= 0.001:
        return f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"
    e = int(np.floor(np.log10(p)))
    return f"p={p / 10 ** e:.1f}\\times10^{{{e}}}"


def rows(summ, keys, sub=None, rot_only=False):
    vals = {k: (summ[k][sub] if sub else summ[k]) for k in keys if k in summ and "sim3_mean" in summ[k]}
    full64 = [k for k in vals if summ[k].get("frames", [64]) == [64] and k != "static"]
    out = []
    if rot_only:
        best = min(vals[k]["rpe_r_mean"] for k in full64)
    else:
        bs, bm = min(vals[k]["sim3_mean"] for k in full64), min(vals[k]["sim3_median"] for k in full64)
    for k in keys:
        if k not in vals:
            continue
        v, name = vals[k], LAB[k] + (f" \\citep{{{CITE[k]}}}" if k in CITE else "")
        if k in ("dust3r", "endo3r", "mast3r"):
            name += "$^{\\dagger}$"
        if k == "heightfield":
            name = "\\textbf{" + name + "}"
        pair = "--"
        if "hf_better_on" in v:
            pair = f"${v['hf_better_on']}/{v.get('paired_n', v.get('n'))}$ (${fmt_p(v['wilcoxon_p'])}$)"
        if rot_only:
            r = f"{v['rpe_r_mean']:.3f}"
            r = f"$\\mathbf{{{r}}}$" if (k in full64 and r == f"{best:.3f}") else f"${r}$"
            out.append(f"{name} & ${v.get('n', summ[k]['tracked'])}$ & {r} & ${v['rpe_r_median']:.3f}$ & {pair} \\\\")
        else:
            m = f"{v['sim3_mean']:.2f}"; md = f"{v['sim3_median']:.2f}"
            m = f"\\mathbf{{{m}}}" if (k in full64 and m == f"{bs:.2f}") else m
            md = f"\\mathbf{{{md}}}" if (k in full64 and md == f"{bm:.2f}") else md
            n = v.get("n", summ[k]["tracked"])
            out.append(f"{name} & ${n}/{n if sub else summ[k]['total']}$ & ${m}$ / ${md}$ & ${v['rpe_t_mean']:.2f}$ & {pair} \\\\")
    return "\n".join(out)


SEGS = [("gtraj_20261002_024309", "segment_04_43s_66s", "S1"), ("gtraj_20261002_024309", "segment_06_85s_187s", "S2")]
SIMS = [("seq_00000267", "M1"), ("seq_00000062", "M2"), ("seq_00001039", "M3")]
COL = {"heightfield": "#0f766e", "vggt": "#ef4444", "reloc3r": "#f59e0b", "cut3r": "#8b5cf6", "endodac": "#6b7280", "afsfm": "#a16207",
       "endofast3r": "#84cc16", "pi3": "#ec4899", "droid": "#0ea5e9"}


def case(sdir, bench_dir, suffix, keys):
    sid = os.path.basename(sdir)
    gt_all = load_pose_txt(os.path.join(sdir, "pose_c2w.txt"))
    n = min(len(list_color_frames(sdir)), len(gt_all))
    gt = gt_all[:n][select_frame_indices(gt_all[:n], "uniform", max_frames=64)]
    tr = {"heightfield": load_pose_txt(str(HERE / "hf_benchmarks" / bench_dir / "heightfield" / f"{sid}_est_c2w.txt"))}
    for k in keys:
        p = SOTA / f"{k}_{suffix}" / f"{sid}_est_c2w.txt"
        if p.is_file():
            e = load_pose_txt(str(p))
            if len(e) == len(gt):
                tr[k] = e
    return gt, tr


def aligned(est, gt):
    a, _ = align_trajectories(est, gt, with_scale=True)
    p, g = a[:, :3, 3], gt[:, :3, 3]
    return p, np.linalg.norm(p - g, axis=1)


def plot3d(ax, gt, tr, title):
    g = gt[:, :3, 3]
    ax.plot(g[:, 0], g[:, 1], g[:, 2], color="black", lw=2.6, label="Ground truth", zorder=10)
    ax.scatter(*g[0], color="#16a34a", marker="o", s=26, zorder=11)
    ax.scatter(*g[-1], color="#dc2626", marker="^", s=30, zorder=11)
    res = {}
    for k, e in tr.items():
        p, err = aligned(e, gt)
        res[k] = float(np.sqrt(np.mean(err ** 2)))
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=COL[k], lw=1.7 if k == "heightfield" else 0.9, label=LAB[k])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("$\\Delta X$ (mm)", labelpad=-2)
    ax.set_ylabel("$\\Delta Y$ (mm)", labelpad=-2)
    ax.set_zlabel("$\\Delta Z$ (mm)", labelpad=-4)
    ax.tick_params(pad=-2)
    return res


def fig11():
    keys_r = ["vggt", "reloc3r", "cut3r", "endodac", "afsfm", "endofast3r", "pi3"]
    keys_s = ["vggt", "reloc3r", "cut3r", "endodac", "afsfm", "endofast3r", "pi3", "droid"]
    fig = plt.figure(figsize=(11.2, 7.4))
    metrics = {}
    cases = []
    for sess, seg, tag in SEGS:
        cases.append((tag, *case(str(ENDOSIM_ROOT / "sim_data" / "egomo_gtraj" / sess / seg), "real18", "egomo_gtraj", keys_r)))
    for sid, tag in SIMS:
        cases.append((tag, *case(str(ENDOSIM_ROOT / "sim_data" / "test" / sid), "sim92", "simtest", keys_s)))
    ax = fig.add_subplot(2, 3, 1, projection="3d"); metrics["S1"] = plot3d(ax, cases[0][1], cases[0][2], "(a) Real Segment S1")
    ax = fig.add_subplot(2, 3, 2, projection="3d"); metrics["S2"] = plot3d(ax, cases[1][1], cases[1][2], "(b) Real Segment S2")
    ax = fig.add_subplot(2, 3, 3)
    for k, e in cases[0][2].items():
        ax.plot(aligned(e, cases[0][1])[1], color=COL[k], lw=1.7 if k == "heightfield" else 0.9)
    ax.set_xlabel("Frame index"); ax.set_ylabel("3D position error (mm)")
    ax.set_title("(c) Frame-wise error, Segment S1", fontweight="bold"); ax.grid(linestyle="--", alpha=0.35)
    for j, (tag, gt, tr) in enumerate(cases[2:]):
        ax = fig.add_subplot(2, 3, 4 + j, projection="3d")
        metrics[tag] = plot3d(ax, gt, tr, f"({chr(100 + j)}) Simulated Sequence {tag}")
    handles = {}
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=6, fontsize=7, frameon=True)
    fig.subplots_adjust(left=0.045, right=0.965, top=0.95, bottom=0.095, wspace=0.28, hspace=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(IMG / f"Fig_3d_trajectories_comparison.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return metrics, cases


def perseq_rows(metrics, cases):
    order = ["vggt", "reloc3r", "cut3r", "endodac", "afsfm", "endofast3r", "pi3", "droid", "heightfield"]
    names = {"S1": "Real Segment S1", "S2": "Real Segment S2", "M1": "Simulated Seq.\\ M1 (free)", "M2": "Simulated Seq.\\ M2 (insertion)",
             "M3": "Simulated Seq.\\ M3 (insertion)"}
    out = []
    for tag, gt, _ in cases:
        path = float(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum())
        m = metrics[tag]
        best = min(m.values())
        cells = []
        for k in order:
            if k not in m:
                cells.append("--")
            else:
                c = f"{m[k]:.2f}"
                cells.append(f"$\\mathbf{{{c}}}$" if abs(m[k] - best) < 1e-9 else f"${c}$")
        out.append(f"{names[tag]} (${path:.1f}\\,\\mathrm{{mm}}$) & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def main():
    fig10()
    r18, s92, c3 = load("real18")["summary"], load("sim92")["summary"], load("c3vd10")["summary"]
    for s in (r18, c3):
        for k, v in s.items():
            if "hf_better_on" in v:
                v["paired_n"] = v.get("paired_n", s["heightfield"]["tracked"])
    order18 = ["vggt", "endodac", "cut3r", "reloc3r", "afsfm", "endofast3r", "pi3", "static", "heightfield"]
    order92 = ["droid", "dust3r", "endo3r", "mast3r", "vggt", "cut3r", "endodac", "afsfm", "reloc3r", "endofast3r", "pi3", "static", "heightfield"]
    orderc3 = ["afsfm", "droid", "endodac", "endofast3r", "pi3", "reloc3r", "static", "heightfield"]
    (HERE / "rows_real18.tex").write_text(rows(r18, order18), encoding="utf-8")
    (HERE / "rows_sim71.tex").write_text(rows(s92, order92, sub="translating"), encoding="utf-8")
    (HERE / "rows_sim21rot.tex").write_text(rows(s92, order92, sub="orbital", rot_only=True), encoding="utf-8")
    (HERE / "rows_c3vd10.tex").write_text(rows(c3, orderc3), encoding="utf-8")
    metrics, cases = fig11()
    (HERE / "rows_perseq.tex").write_text(perseq_rows(metrics, cases), encoding="utf-8")
    (HERE / "per_sequence_trajs_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
