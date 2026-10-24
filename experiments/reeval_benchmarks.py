#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-evaluate every method on its own frame set with one metric implementation.

Methods with 64 poses are scored on the 64 uniformly sampled frames, methods that were run on 32 frames
(memory-limited pairwise models) on the 32 uniformly sampled frames; partial trajectories (ORB-SLAM3) are
reported from the evaluation harness and excluded from paired tests. Sim92 results are also split by the
simulator's motion type (insertion, free, orbital).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
from paths import ENDOSIM_ROOT  # noqa: E402
sys.path.insert(0, str(ENDOSIM_ROOT))
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from paths import eval_with_scale_correction  # noqa: E402

SOTA = ENDOSIM_ROOT / "results" / "sota"
LISTS = ENDOSIM_ROOT / "lists"
B = {
    "real18": ("egomo_gtraj.txt", "egomo_gtraj", ["vggt", "endodac", "cut3r", "reloc3r", "afsfm", "endofast3r", "pi3"]),
    "sim92": ("simtest92.txt", "simtest", ["droid", "dust3r", "endo3r", "mast3r", "vggt", "cut3r", "endodac", "afsfm", "reloc3r", "endofast3r", "pi3"]),
    "c3vd10": ("c3vd_real.txt", "c3vd_u64", ["afsfm", "droid", "endodac", "endofast3r", "pi3", "reloc3r"]),
}


def met(est, gt):
    r = eval_with_scale_correction(est, gt)
    return {"sim3": float(r["ate_sim3"]["rmse"]), "rpe_t": float(r["rpe_1"]["trans_mm_mean"]), "rpe_r": float(r["rpe_1"]["rot_deg_mean"])}


def run(name):
    lst, suffix, bases = B[name]
    seqs = [ln.strip() for ln in open(LISTS / lst, encoding="utf-8") if ln.strip()]
    recs = {k: {} for k in ["heightfield", "static"] + bases}
    nfr = {k: set() for k in bases}
    motion = {}
    for sdir in seqs:
        sid = os.path.basename(sdir.rstrip("/\\"))
        gt_all = load_pose_txt(os.path.join(sdir, "pose_c2w.txt"))
        n = min(len(list_color_frames(sdir)), len(gt_all))
        gt_all = gt_all[:n]
        gts = {64: gt_all[select_frame_indices(gt_all, "uniform", max_frames=64)],
               32: gt_all[select_frame_indices(gt_all, "uniform", max_frames=32)]}
        mp = os.path.join(sdir, "meta.json")
        motion[sid] = json.load(open(mp, encoding="utf-8")).get("motion_type") if os.path.exists(mp) else None
        hf = load_pose_txt(str(HERE / "hf_benchmarks" / name / "heightfield" / f"{sid}_est_c2w.txt"))
        recs["heightfield"][sid] = met(hf, gts[64])
        recs["static"][sid] = met(np.tile(np.eye(4), (len(gts[64]), 1, 1)), gts[64])
        for k in bases:
            p = SOTA / f"{k}_{suffix}" / f"{sid}_est_c2w.txt"
            if not p.is_file():
                continue
            e = load_pose_txt(str(p))
            for nf, g in gts.items():
                short = len(g) == n and len(e) > len(g)  # sequence shorter than the protocol: harness truncates
                if (len(e) == len(g) or short) and np.isfinite(e).all():
                    recs[k][sid] = met(e[:len(g)], g)
                    nfr[k].add(nf)
                    break
    ids = list(recs["heightfield"].keys())
    summ = {}
    for k, d in recs.items():
        s = {"tracked": len(d), "total": len(ids)}
        if d:
            sim = np.array([v["sim3"] for v in d.values()])
            s.update({"sim3_mean": float(sim.mean()), "sim3_median": float(np.median(sim)),
                      "rpe_t_mean": float(np.mean([v["rpe_t"] for v in d.values()])),
                      "rpe_r_mean": float(np.mean([v["rpe_r"] for v in d.values()]))})
        if k in nfr:
            s["frames"] = sorted(nfr[k])
        if k != "heightfield" and d and (k == "static" or nfr.get(k) == {64}) and len(d) == len(ids):
            a = np.array([recs["heightfield"][q]["sim3"] for q in ids])
            c = np.array([d[q]["sim3"] for q in ids])
            s.update({"hf_better_on": int((a < c).sum()), "paired_n": len(ids), "wilcoxon_p": float(stats.wilcoxon(a - c).pvalue)})
        if name == "sim92" and d:
            s["by_motion"] = {}
            for mt in ("insertion", "free", "orbital"):
                v = [d[q]["sim3"] for q in ids if motion.get(q) == mt and q in d]
                rr = [d[q]["rpe_r"] for q in ids if motion.get(q) == mt and q in d]
                if v:
                    s["by_motion"][mt] = {"n": len(v), "sim3_mean": float(np.mean(v)), "sim3_median": float(np.median(v)),
                                          "rpe_r_mean": float(np.mean(rr))}
            tq = [q for q in ids if motion.get(q) != "orbital" and q in d]
            oq = [q for q in ids if motion.get(q) == "orbital" and q in d]
            s["translating"] = {"n": len(tq), "sim3_mean": float(np.mean([d[q]["sim3"] for q in tq])),
                                "sim3_median": float(np.median([d[q]["sim3"] for q in tq])),
                                "rpe_t_mean": float(np.mean([d[q]["rpe_t"] for q in tq])), "rpe_r_mean": float(np.mean([d[q]["rpe_r"] for q in tq]))}
            s["orbital"] = {"n": len(oq), "rpe_r_mean": float(np.mean([d[q]["rpe_r"] for q in oq])),
                            "rpe_r_median": float(np.median([d[q]["rpe_r"] for q in oq]))}
            if k != "heightfield" and (k == "static" or nfr.get(k) == {64}) and len(d) == len(ids):
                a = np.array([recs["heightfield"][q]["sim3"] for q in tq]); c = np.array([d[q]["sim3"] for q in tq])
                s["translating"].update({"hf_better_on": int((a < c).sum()), "wilcoxon_p": float(stats.wilcoxon(a - c).pvalue)})
                a = np.array([recs["heightfield"][q]["rpe_r"] for q in oq]); c = np.array([d[q]["rpe_r"] for q in oq])
                s["orbital"].update({"hf_better_on": int((a < c).sum()), "wilcoxon_p": float(stats.wilcoxon(a - c).pvalue)})
        summ[k] = s
    if name == "sim92":
        h = json.load(open(SOTA / "orbslam3_simtest" / "summary.json", encoding="utf-8"))
        hs = h.get("summary", h)
        summ["orbslam3_harness"] = {"tracked": hs.get("n_seq"), "failed": hs.get("n_failed"), "sim3_mean_on_tracked": hs.get("ate_sim3_rmse_mean"),
                                    "sim3_median_on_tracked": hs.get("ate_sim3_rmse_median"), "note": "partial trajectories; harness protocol"}
    out = {"doc": __doc__.strip().splitlines()[0], "summary": summ, "per_sequence": recs, "motion": motion}
    (HERE / "hf_benchmarks" / f"{name}_reeval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"===== {name}")
    for k, s in summ.items():
        if "sim3_mean" in s:
            line = f"{k:12s} {s['tracked']:3d}/{s['total']:3d} sim3 {s['sim3_mean']:7.2f}/{s['sim3_median']:6.2f} rpe_t {s['rpe_t_mean']:5.2f} rpe_r {s['rpe_r_mean']:6.3f}"
            if "frames" in s:
                line += f" frames {s['frames']}"
            if "hf_better_on" in s:
                line += f" | HF better {s['hf_better_on']}/{s['paired_n']} p {s['wilcoxon_p']:.2g}"
            if "translating" in s:
                t_, o_ = s["translating"], s["orbital"]
                line += (f"\n      translating n={t_['n']} sim3 {t_['sim3_mean']:.2f}/{t_['sim3_median']:.2f} rpe_r {t_['rpe_r_mean']:.3f}"
                         + (f" HFbetter {t_['hf_better_on']} p {t_['wilcoxon_p']:.2g}" if "hf_better_on" in t_ else "")
                         + f" | orbital n={o_['n']} rpe_r {o_['rpe_r_mean']:.3f}/{o_['rpe_r_median']:.3f}"
                         + (f" HFbetter {o_['hf_better_on']} p {o_['wilcoxon_p']:.2g}" if "hf_better_on" in o_ else ""))
            print(line)
        else:
            print(k, s)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["real18", "sim92", "c3vd10"]):
        run(nm)
