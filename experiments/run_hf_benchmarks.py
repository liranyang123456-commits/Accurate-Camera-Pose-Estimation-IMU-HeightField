#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Height-field tracker vs. baselines on the 18 real segments, 92 simulated sequences, and 10 C3VD sequences.

Protocol (identical for every method): at most 64 frames sampled uniformly per sequence
(select_frame_indices(..., "uniform", 64)), Sim(3) Umeyama ATE RMSE, and consecutive-frame RPE after the
Sim(3) scale (eval_with_scale_correction). Baselines are re-evaluated from their stored trajectories on the
same frames. A static (identity) trajectory is included as a reference.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import ENDOSIM_ROOT  # noqa: E402
sys.path.insert(0, str(ENDOSIM_ROOT))
from hf_tracker import CFG, static_mask, track  # noqa: E402
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from paths import eval_with_scale_correction  # noqa: E402

SOTA = ENDOSIM_ROOT / "results" / "sota"
LISTS = ENDOSIM_ROOT / "lists"
OUT = HERE / "hf_benchmarks"

BENCH = {
    "real18": dict(list="egomo_gtraj.txt", suffix="egomo_gtraj", mask=False,
                   baselines=["vggt", "endodac", "cut3r", "reloc3r", "afsfm", "endofast3r", "pi3", "mast3r", "orbslam3"]),
    "sim92": dict(list="simtest92.txt", suffix="simtest", mask=False,
                  baselines=["orbslam3", "droid", "dust3r", "endo3r", "mast3r", "vggt", "cut3r", "endodac", "afsfm", "reloc3r", "endofast3r", "pi3"]),
    "c3vd10": dict(list="c3vd_real.txt", suffix="c3vd_u64", mask=True,
                   baselines=["afsfm", "droid", "endodac", "endofast3r", "pi3", "reloc3r"]),
}


def ref_fraction(sdir):
    p = os.path.join(sdir, "meta.json")
    if not os.path.exists(p):
        return None
    refs = [r for r in json.load(open(p, encoding="utf-8")).get("reference_fraction", []) if r is not None]
    return float(np.mean(refs)) if refs else None


def metrics(est, gt):
    m = min(len(est), len(gt))
    r = eval_with_scale_correction(est[:m], gt[:m])
    return {"sim3": float(r["ate_sim3"]["rmse"]), "rpe_t": float(r["rpe_1"]["trans_mm_mean"]), "rpe_r": float(r["rpe_1"]["rot_deg_mean"]), "n": int(m)}


def run(name):
    b = BENCH[name]
    seqs = [ln.strip() for ln in open(LISTS / b["list"], encoding="utf-8") if ln.strip()]
    odir = OUT / name / "heightfield"
    odir.mkdir(parents=True, exist_ok=True)
    recs = {"heightfield": {}, "static": {}}
    for k in b["baselines"]:
        recs[k] = {}
    fit_all = []
    for sdir in seqs:
        sid = os.path.basename(sdir.rstrip("/\\"))
        frames = list_color_frames(sdir)
        gt_all = load_pose_txt(os.path.join(sdir, "pose_c2w.txt"))
        n = min(len(frames), len(gt_all))
        idx = select_frame_indices(gt_all[:n], "uniform", max_frames=64)
        gt = gt_all[:n][idx]
        imgs = [cv2.imread(frames[i]) for i in idx]
        mask = static_mask(imgs) if b["mask"] else None
        res = track(imgs, mask)
        est = res["poses"]
        fit_all.append(res["fitness"])
        np.savetxt(odir / f"{sid}_est_c2w.txt", est.reshape(len(est), 16), fmt="%.6f")
        recs["heightfield"][sid] = metrics(est, gt)
        recs["heightfield"][sid]["ref_frac"] = ref_fraction(sdir)
        recs["static"][sid] = metrics(np.tile(np.eye(4), (len(gt), 1, 1)), gt)
        for k in b["baselines"]:
            p = SOTA / f"{k}_{b['suffix']}" / f"{sid}_est_c2w.txt"
            if p.is_file():
                try:
                    e = load_pose_txt(str(p))
                    if len(e) >= max(3, int(0.9 * len(gt))) and np.isfinite(e).all():
                        recs[k][sid] = metrics(e, gt)
                except Exception:
                    pass
        print(f"[{name}] {sid}: HF sim3 {recs['heightfield'][sid]['sim3']:.2f} rpe_r {recs['heightfield'][sid]['rpe_r']:.3f} "
              f"| static {recs['static'][sid]['sim3']:.2f}", flush=True)
    seq_ids = list(recs["heightfield"].keys())
    summ = {}
    for k, d in recs.items():
        if not d:
            summ[k] = {"tracked": 0, "total": len(seq_ids)}
            continue
        sim = np.array([v["sim3"] for v in d.values()])
        s = {"tracked": len(d), "total": len(seq_ids), "sim3_mean": float(sim.mean()), "sim3_median": float(np.median(sim)),
             "rpe_t_mean": float(np.mean([v["rpe_t"] for v in d.values()])), "rpe_r_mean": float(np.mean([v["rpe_r"] for v in d.values()]))}
        common = [q for q in seq_ids if q in d]
        if k != "heightfield" and len(common) >= 5:
            a = np.array([recs["heightfield"][q]["sim3"] for q in common])
            c = np.array([d[q]["sim3"] for q in common])
            s["hf_better_on"] = int((a < c).sum())
            s["paired_n"] = len(common)
            s["wilcoxon_p_sim3"] = float(stats.wilcoxon(a - c).pvalue) if np.any(a != c) else 1.0
        summ[k] = s
    if name == "sim92":
        strata = {"low": [], "mid": [], "high": []}
        for q, v in recs["heightfield"].items():
            rf = v.get("ref_frac")
            key = "low" if (rf is not None and rf < 0.3) else ("mid" if (rf is not None and rf < 0.7) else "high")
            strata[key].append(v["sim3"])
        summ["heightfield_strata"] = {k: {"n": len(v), "sim3_mean": float(np.mean(v))} for k, v in strata.items()}
    fit = np.concatenate(fit_all)
    summ["heightfield_fitness"] = {"mean": float(fit.mean()), "lt_0.2": int((fit < 0.2).sum()), "pairs": int(len(fit))}
    out = {"protocol": __doc__.strip().splitlines()[0], "config": CFG, "summary": summ, "per_sequence": recs}
    (OUT / f"{name}_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["c3vd10", "real18", "sim92"]):
        run(nm)
