#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure and table rows of the controlled fusion study (eval_controlled_fusion_sim.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
IMG = HERE / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.6, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8})
COL = {"0.1": "#0f766e", "0.5": "#2563eb", "1.0": "#d97706", "2.0": "#b91c1c"}
TITLE = {"sim92": "(a) 92 simulated sequences", "real18": "(b) 18 real 4K-referenced segments"}


def main():
    d = json.loads((HERE / "controlled_fusion_sim_results.json").read_text(encoding="utf-8"))
    grid = [0.0] + list(d["gamma_grid"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, name in zip(axes, ("sim92", "real18")):
        res = d[name]["results"]
        for sr, r in res.items():
            y = [r["imu"]["rot_all"][0]] + [r[f"g{g}_t{d['tau_deg']}"]["rot_all"][0] for g in d["gamma_grid"]]
            ax.plot(grid, y, color=COL[sr], lw=1.4, marker="o", ms=2.5, label=f"$\\sigma_R={sr}^\\circ$")
            gs, ys = r["gamma_star_gated"][0], r["gstar_gated"]["rot_all"][0]
            gu, yu = r["gamma_star_ungated"][0], r["gstar_ungated"]["rot_all"][0]
            ax.plot([gs], [ys], marker="*", ms=9, color=COL[sr], mec="black", mew=0.5, ls="none")
            ax.plot([gu], [yu], marker="D", ms=4.5, mfc="white", color=COL[sr], mew=1.0, ls="none")
        ax.axhline(d[name]["visual_only"]["rot_all"], color="gray", ls="--", lw=0.9)
        ax.text(0.5, d[name]["visual_only"]["rot_all"] + 0.03, "height field alone", ha="center", va="bottom", fontsize=6.5, color="gray")
        ax.set_title(TITLE[name], fontweight="bold")
        ax.set_xlabel("blend weight $\\gamma$ ($\\tau_{\\mathrm{rot}}=3^\\circ$; $\\gamma=0$: inertial only)")
        ax.set_ylabel("RPE$_1$ rotation error (deg)")
        ax.grid(ls="--", alpha=0.35)
    h, lab = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="*", ms=8, color="gray", mec="black", ls="none"),
          plt.Line2D([], [], marker="D", ms=4.5, mfc="white", color="gray", ls="none")]
    lab += ["$\\gamma^\\star$ (gated variance)", "$\\gamma^\\star$ (all increments)"]
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(IMG / f"Fig_controlled_fusion.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")

    rows = []
    for name, lab in (("sim92", "Simulated (92)"), ("real18", "Real (18)")):
        res = d[name]["results"]
        for i, (sr, r) in enumerate(res.items()):
            f = lambda k, m="rot_all", nd=2: f"{r[k][m][0]:.{nd}f}"  # noqa: E731
            best = min(d["gamma_grid"], key=lambda g: r[f"g{g}_t{d['tau_deg']}"]["rot_all"][0])
            rows.append(
                f"{lab if i == 0 else ''} & ${sr}$ & ${f('imu')}$ & ${f('g0.5_t3.0')}$ & ${f('g0.5_tinf')}$ & "
                f"${f('gstar_gated')}$ ($\\gamma^\\star={r['gamma_star_gated'][0]:.2f}$) & ${best:.1f}$ & "
                f"${100 * r['g0.5_t3.0']['accepted'][0]:.0f}$ & ${r['g0.5_t3.0']['max_excess_minus_bound'][0] + 1.5:.2f}$ & "
                f"${f('imu', 'ate_trans')}$ / ${f('g0.5_t3.0', 'ate_trans')}$ / ${f('g0.5_rot_only', 'ate_trans')}$ \\\\")
        rows.append("\\midrule" if name == "sim92" else "")
    (HERE / "rows_controlled_fusion.tex").write_text("\n".join(r for r in rows if r), encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
