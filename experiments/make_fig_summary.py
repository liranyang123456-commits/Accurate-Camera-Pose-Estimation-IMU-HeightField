"""Summary figure: session-level paired gains of the gated fusion (forest plots) and the error of the proposed
estimators relative to the zero-motion / static reference on every benchmark with such a reference.

Inputs are the result files that also produce the tables: fusion_hybrid_results.json and
review_addenda_results.json (11 held-out sessions), fixedtest_aligned_results.json (zero motion),
miti_fusion_results.json (nine MITI sequences), and the rows_*.tex files of the visual benchmarks.
Output: experiments/figures/Fig_summary.pdf and .png.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def parse_rows(name):
    """Return {method: (ate_mean_or_rot_mean)} from a rows_*.tex file."""
    rows = {}
    for line in (HERE / name).read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 3:
            continue
        label = re.sub(r"\\cite[a-z]*\{[^}]*\}", "", cells[0])
        label = re.sub(r"\$\^\{\\dagger\}\$", "", label)
        label = re.sub(r"\\textbf\{([^}]*)\}", r"\1", label).strip().replace("Endo-FASt3R", "Endo-FASt3r")
        nums = re.findall(r"[-+]?\d+\.\d+", cells[2].replace(r"\mathbf", ""))
        if nums:
            rows[label] = float(nums[0])
    return rows


def main():
    hyb = load("fusion_hybrid_results.json")
    add = load("review_addenda_results.json")
    fixed = load("fixedtest_aligned_results.json")
    miti = load("miti_fusion_results.json")
    st, holm = hyb["stats"], add["holm_table_session_stats"]

    fig = plt.figure(figsize=(5.6, 9.2))
    # left, bottom, width, height in figure fractions: forest axes occupy the left part, annotations run to the right
    pos_a = [0.34, 0.735, 0.22, 0.215]
    pos_b = [0.34, 0.445, 0.22, 0.195]
    pos_c = [0.26, 0.05, 0.70, 0.28]

    def forest(ax, keys, labels, unit, title, xmax):
        y = list(range(len(keys)))[::-1]
        for yi, k, lab in zip(y, keys, labels):
            s = st[k]
            lo, hi = s["ci95"]
            color = "#1b7f79" if lo > 0 else ("#b5452a" if hi < 0 else "#6b6b6b")
            ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, solid_capstyle="round")
            ax.plot(s["gain"], yi, "o", color=color, ms=6.5)
            ax.text(1.07, yi, f"{s['gain']:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=6.6, color="#333333", clip_on=False, transform=ax.get_yaxis_transform())
            ax.text(2.25, yi, f"{s['sessions_better']}/{s['n_sessions']}", va="center", fontsize=6.6, color="#333333", clip_on=False, transform=ax.get_yaxis_transform())
            ax.text(2.62, yi, f"{holm[k]['p_holm']:.3f}", va="center", fontsize=6.6, color="#333333", clip_on=False, transform=ax.get_yaxis_transform())
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.8)
        ax.set_xlim(min(-0.05 * xmax, min(st[k]["ci95"][0] for k in keys) * 1.15), xmax)
        ax.set_xlabel(f"Gain of A over B ({unit}); positive favors A", fontsize=7)
        ax.set_title(title, fontsize=8, loc="left", pad=18)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for x, head in ((1.07, f"gain ({unit}) [95% interval]"), (2.25, "A better"), (2.62, "Holm $p$")):
            ax.text(x, len(keys) - 0.62, head, fontsize=6.6, color="#333333", clip_on=False, va="bottom", fontweight="bold", transform=ax.get_yaxis_transform())

    rot_keys = ["inertial_vs_orbvo_rot", "inertial_vs_hvo_rot", "vo_validation_rot", "hvo_validation_rot", "hf_validation_rot",
                "vo_default_rot", "hvo_default_rot", "hf_default_rot"]
    rot_labels = ["Inertial vs. ORB-VO", "Inertial vs. ORB-HVO", "ORB-VO blend (val. w.) vs. inertial", "ORB-HVO blend (val. w.) vs. inertial",
                  "HF blend (val. w.) vs. inertial", r"ORB-VO blend ($\gamma$=0.5) vs. inertial", r"ORB-HVO blend ($\gamma$=0.5) vs. inertial",
                  r"HF blend ($\gamma$=0.5) vs. inertial"]
    tr_keys = ["vo_validation_trans", "hvo_validation_trans", "hf_validation_trans", "vo_default_trans", "hvo_default_trans", "hf_default_trans",
               "slam_default_trans"]
    tr_labels = ["ORB-VO blend (val. w.) vs. inertial", "ORB-HVO blend (val. w.) vs. inertial", "HF blend (val. w.) vs. inertial",
                 r"ORB-VO blend ($\gamma$=0.5) vs. inertial", r"ORB-HVO blend ($\gamma$=0.5) vs. inertial", r"HF blend ($\gamma$=0.5) vs. inertial",
                 r"ORB-SLAM3 blend ($\gamma$=0.5) vs. inertial"]
    ax1 = fig.add_axes(pos_a)
    forest(ax1, rot_keys, rot_labels, "deg", "(a) Rotation, 11 held-out sessions", 4.8)
    ax2 = fig.add_axes(pos_b)
    forest(ax2, tr_keys, tr_labels, "mm", "(b) Translation, 11 held-out sessions", 4.2)

    # (c) error relative to the zero-motion / static reference
    zero = fixed["rows"]["zero_motion"]
    rows_h = hyb["rows"]
    real, sim, c3vd, rot21 = parse_rows("rows_real18.tex"), parse_rows("rows_sim71.tex"), parse_rows("rows_c3vd10.tex"), parse_rows("rows_sim21rot.tex")

    def best(rows):
        cand = {k: v for k, v in rows.items() if k not in ("Static (reference)", "Ours (height field)")}
        k = min(cand, key=cand.get)
        return k, cand[k]

    items = [
        ("11 sessions, rotation\n(N=2312, recorded IMU)", rows_h["hvo_validation"]["rot_mean"] / zero["rot_mean"], "gated ORB-HVO blend",
         rows_h["hvo_only"]["rot_mean"] / zero["rot_mean"], "ORB-HVO alone"),
        ("11 sessions, translation\n(N=2312, recorded IMU)", rows_h["hvo_validation"]["trans_mean"] / zero["trans_mean"], "gated ORB-HVO blend (inertial scale)",
         rows_h["hvo_only"]["trans_mean"] / zero["trans_mean"], "ORB-HVO alone (reference scale)"),
        ("MITI, rotation\n(N=3180, recorded IMU)", miti["rows"]["hf_rig_weights_tlio"]["rot_mean"] / miti["rows"]["zero_motion"]["rot_mean"], "gated HF blend",
         miti["rows"]["hf_only"]["rot_mean"] / miti["rows"]["zero_motion"]["rot_mean"], "height field alone"),
        ("MITI, translation\n(N=3180, recorded IMU)", miti["rows"]["hf_rig_weights_tlio"]["trans_mean"] / miti["rows"]["zero_motion"]["trans_mean"],
         "gated HF blend (TLIO translation)", miti["rows"]["hf_only"]["trans_mean"] / miti["rows"]["zero_motion"]["trans_mean"], "height field alone"),
        ("18 real segments, ATE\n(visual only)", real["Ours (height field)"] / real["Static (reference)"], "height field",
         best(real)[1] / real["Static (reference)"], best(real)[0]),
        ("71 simulated, ATE\n(visual only)", sim["Ours (height field)"] / sim["Static (reference)"], "height field",
         best(sim)[1] / sim["Static (reference)"], best(sim)[0]),
        ("21 pure rotation, RPE rot.\n(visual only)", rot21["Ours (height field)"] / rot21["Static (reference)"], "height field",
         best(rot21)[1] / rot21["Static (reference)"], best(rot21)[0]),
        ("10 C3VD, ATE\n(visual only)", c3vd["Ours (height field)"] / c3vd["Static (reference)"], "height field",
         best(c3vd)[1] / c3vd["Static (reference)"], best(c3vd)[0]),
    ]
    ax3 = fig.add_axes(pos_c)
    y = list(range(len(items)))[::-1]
    h = 0.38
    for yi, (lab, ours, ours_name, other, other_name) in zip(y, items):
        ax3.barh(yi + h / 2, ours, height=h, color="#1b7f79", label="proposed estimator" if yi == y[0] else None)
        ax3.barh(yi - h / 2, other, height=h, color="#4d7fc4", label="best compared method with full coverage" if yi == y[0] else None)
        ax3.text(ours + 0.015, yi + h / 2, f"{ours:.2f}  {ours_name}", va="center", fontsize=6.3)
        ax3.text(other + 0.015, yi - h / 2, f"{other:.2f}  {other_name}", va="center", fontsize=6.3)
    ax3.axvline(1.0, color="k", lw=0.9, ls="--")
    ax3.text(1.0, len(items) - 0.4, "zero motion / static reference = 1", fontsize=6.8, ha="center", va="bottom")
    ax3.set_yticks(y)
    ax3.set_yticklabels([it[0] for it in items], fontsize=6.8)
    ax3.set_xlim(0, 1.9)
    ax3.set_ylim(-1.7, len(items) - 0.1)
    ax3.set_xlabel("Mean error / reference error on the same windows or sequences (lower is better)",
                   fontsize=7)
    ax3.set_title("(c) Proposed estimator and best compared method relative to the reference", fontsize=8,
                  loc="left")
    ax3.legend(fontsize=6.8, loc="lower right", frameon=False, ncol=1)
    ax3.tick_params(axis="x", labelsize=6.8)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)

    fig.savefig(OUT / "Fig_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig_summary.png", bbox_inches="tight", dpi=200)
    for lab, ours, ours_name, other, other_name in items:
        print(f"{lab.splitlines()[0]:32s} ours {ours:.3f} ({ours_name}) | other {other:.3f} ({other_name})")
    print("wrote", OUT / "Fig_summary.pdf")


if __name__ == "__main__":
    main()
