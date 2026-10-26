#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inertial, visual, and gated visual-inertial relative pose on the 11 held-out sessions after the time alignment.

GRU6D is the model trained on the time-aligned windows (train_inertial_baselines.py --ds ..._offset-0.50); seed 0 is
the evaluated model and seeds 1-2 are reported as a robustness range. Window pool, reference interpolation, relative
pose convention, gate, blend, and the two translation scales are those of the earlier evaluation. Visual increments
use camera timestamps and are unaffected by the alignment.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation, Slerp

from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from eval_vio_fixedtest_aggregate import _load_fixed_test_sessions_from_train_config  # noqa: E402
from eval_vio_trajectory_rpe_ate import (  # noqa: E402
    PoseSeq, _interp_pose, _load_gt_poses_from_frames_labeled, _load_traj_tum, _rel_end_frame)
from run_classical_vo_vio_fixedtest import _estimate_scale_from_gt  # noqa: E402
from eval_best_model_metrics import _project_to_so3_np  # noqa: E402
from build_fixed_seq_dataset_from_session import load_imu_stream_features  # noqa: E402

import eval_gyro_time_alignment as ga  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = IMUCAM_ROOT
RUN = ROOT / "out_train_fixed_seq_20260310_031242"
DS = "fixed_seq_len60_win0.500s_delta_window"
VIS = {"vo": ROOT / "vio_runs_classical" / "orb_vo", "slam": ROOT / "vio_runs" / "ORB-SLAM3-mono", "hf": HERE / "hf_sessions"}
PRED = Path(os.environ.get("GRU6D_PRED_DIR", HERE / "inertial_baselines_aligned"))
SEEDS = (0, 1, 2)
TAU, GAMMA, GAMMA_SLAM = 3.0, 0.5, 0.7
OUT = Path(os.environ.get("FIXEDTEST_OUT", HERE / "fixedtest_aligned_results.json"))


def geo(R1, R2):
    return float(np.degrees(Rotation.from_matrix(R1.T @ R2).magnitude()))


def slerp(Ra, Rb, w):
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb], 0)))([w]).as_matrix()[0]


def load_vis(path, gt_seq):
    if not path.exists() or path.stat().st_size == 0:
        return None, 1.0
    est = _load_traj_tum(str(path), time_unit="s", pos_scale=1.0, pose_mode="T_WC")
    return est, float(_estimate_scale_from_gt(est, gt_seq, window_s=0.5))


def build_records():
    al = json.loads((HERE / "gyro_time_alignment_results.json").read_text(encoding="utf-8"))
    vg = al["variants"]["gyro_corrected_convention_offset"]
    d_g, X_g = float(vg["delta_s"]), np.asarray(vg["R_CI"])
    preds = {s: np.load(PRED / f"pred_gru6d_s{s}.npz") for s in SEEDS if (PRED / f"pred_gru6d_s{s}.npz").exists()}
    recs = []
    for sd in map(Path, _load_fixed_test_sessions_from_train_config(str(RUN / "train_config.json"))):
        y = np.load(sd / DS / "y.npy")
        meta = list(csv.DictReader(open(sd / DS / "meta.csv", encoding="utf-8")))
        gt_seq = _load_gt_poses_from_frames_labeled(str(sd / "frames_labeled.csv"), max_reproj_rmse=2.0,
                                                    chessboard_mode="majority", chessboard_name="", pos_scale=1.0)
        t0 = float(gt_seq.t[0])
        vis = {k: load_vis(p / sd.name / "traj_tum.txt", gt_seq) for k, p in VIS.items()}
        t, feat, _ = load_imu_stream_features(str(sd / "imu_stream.csv"))
        track = ga.gyro_track(t, feat[:, 3:6])
        ts_all = np.array([float(m["t_start"]) for m in meta])
        te_all = np.array([float(m["t_end"]) for m in meta])
        B = ga.body_increment(track, ts_all, te_all, d_g)
        for k, m in enumerate(meta):
            t_end = float(m["t_end"])
            t_start = t_end - 0.5
            r = {"session": sd.name, "R_gt": _project_to_so3_np(y[k, :9]), "t_gt": y[k, 9:12].astype(np.float64),
                 "R_gyro": X_g @ B[k].T @ X_g.T,
                 "gru": {s: (_project_to_so3_np(p[sd.name][k, :9]), p[sd.name][k, 9:12].astype(np.float64)) for s, p in preds.items()}}
            ok_ref = t_start >= t0 and _interp_pose(gt_seq, t_end, max_gap_s=0.2) is not None \
                and _interp_pose(gt_seq, t_start, max_gap_s=0.2) is not None
            for key, (seq, scale) in vis.items():
                r[key + "_ok"] = False
                if seq is None or not ok_ref:
                    continue
                e, s0 = _interp_pose(seq, t_end, max_gap_s=0.2), _interp_pose(seq, t_start, max_gap_s=0.2)
                if e is not None and s0 is not None:
                    R, tv = _rel_end_frame(e[0], e[1], s0[0], s0[1])
                    r.update({key + "_ok": True, "R_" + key: R, "t_" + key: tv.reshape(3), "s_" + key: scale})
            recs.append(r)
    return recs, sorted(preds)


def inertial_scales(recs, key, seed):
    out = {}
    for s in dict.fromkeys(r["session"] for r in recs):
        rat = [np.linalg.norm(r["gru"][seed][1]) / np.linalg.norm(r["t_" + key]) for r in recs
               if r["session"] == s and r[key + "_ok"] and np.linalg.norm(r["t_" + key]) > 1e-9]
        out[s] = float(np.median(rat)) if rat else 1.0
    return out


def stream(recs, seed, mode, key=None, tau=TAU, gamma=GAMMA, tmode="ref", scales=None):
    rot, tr, used = [], [], 0
    for r in recs:
        R, t = r["gru"][seed]
        if mode == "gyro":
            R, t = r["R_gyro"], np.zeros(3)
        elif mode in ("blend", "gated") and r[key + "_ok"] and (mode == "blend" or geo(r["R_" + key], R) <= tau):
            used += 1
            R = r["R_" + key] if gamma >= 1.0 else slerp(R, r["R_" + key], gamma)
            if tmode != "rot_only":
                s = r["s_" + key] if tmode == "ref" else scales[r["session"]]
                t = gamma * r["t_" + key] * s + (1 - gamma) * t
        rot.append(geo(R, r["R_gt"]))
        tr.append(float(np.linalg.norm(t - r["t_gt"])))
    return np.asarray(rot), np.asarray(tr), used


def visual_only(recs, key):
    m = np.array([r[key + "_ok"] for r in recs])
    rot = np.array([geo(r["R_" + key], r["R_gt"]) if r[key + "_ok"] else np.nan for r in recs])
    tr = np.array([np.linalg.norm(r["t_" + key] * r["s_" + key] - r["t_gt"]) if r[key + "_ok"] else np.nan for r in recs])
    return rot, tr, m


def summ(r, t, used=None):
    r, t = r[~np.isnan(r)], t[~np.isnan(t)]
    out = {"N": int(r.size), "rot_mean": float(r.mean()), "rot_p50": float(np.median(r)), "rot_p95": float(np.percentile(r, 95)),
           "trans_mean": float(t.mean()), "trans_p50": float(np.median(t)), "trans_p95": float(np.percentile(t, 95)),
           "hit_rot5": float(np.mean(r <= 5.0)), "hit_trans20": float(np.mean(t <= 20.0))}
    if used is not None:
        out["n_visual_used"] = int(used)
    return out


def session_stats(diff, sess, names, B=5000, seed=7):
    groups = [diff[sess == s] for s in names]
    groups = [g[~np.isnan(g)] for g in groups]
    keep = [g for g in groups if g.size]
    per = np.array([g.mean() for g in keep])
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([keep[i] for i in rng.integers(0, len(keep), len(keep))]).mean() for _ in range(B)]
    return {"gain": float(np.concatenate(keep).mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "sessions_better": int((per > 0).sum()), "n_sessions": len(names),
            "p": float(stats.wilcoxon(per).pvalue) if np.any(per != 0) else 1.0}


def main():
    recs, seeds = build_records()
    sess = np.array([r["session"] for r in recs])
    names = list(dict.fromkeys(sess.tolist()))
    out = {"protocol": __doc__.strip().splitlines()[0], "seeds": seeds, "N": len(recs), "rows": {}, "stats": {},
           "sensitivity": [], "bound": {}, "seed_range": {}}
    zero_rot = np.array([geo(np.eye(3), r["R_gt"]) for r in recs])
    zero_t = np.array([np.linalg.norm(r["t_gt"]) for r in recs])
    out["rows"]["zero_motion"] = summ(zero_rot, zero_t)
    s0 = seeds[0]
    sc = {k: inertial_scales(recs, k, s0) for k in VIS}
    rg, tg, _ = stream(recs, s0, "gru")
    out["rows"]["gru6d"] = summ(rg, tg)
    ry, ty, _ = stream(recs, s0, "gyro")
    out["rows"]["gyro_zeroT"] = summ(ry, ty)
    for k in VIS:
        rv, tv, m = visual_only(recs, k)
        out["rows"][f"{k}_only"] = summ(rv, tv)
        out["rows"][f"{k}_only"]["coverage"] = float(m.mean())
    cfg = {"slam_blend": ("blend", "slam", GAMMA_SLAM), "slam_switch": ("blend", "slam", 1.0),
           "vo_gated": ("gated", "vo", GAMMA), "hf_gated": ("gated", "hf", GAMMA),
           "vo_nogate": ("blend", "vo", GAMMA), "hf_nogate": ("blend", "hf", GAMMA)}
    res = {}
    for name, (mode, key, gam) in cfg.items():
        for tmode in ("ref", "inertial", "rot_only"):
            r, t, u = stream(recs, s0, mode, key, gamma=gam, tmode=tmode, scales=sc[key])
            res[(name, tmode)] = (r, t)
            out["rows"][f"{name}_{tmode}"] = summ(r, t, u)
        r = res[(name, "ref")][0]
        out["bound"][name] = {"gamma": gam, "tau": TAU if mode == "gated" else None,
                              "bound_deg": gam * TAU if mode == "gated" else None, "max_excess_deg": float((r - rg).max())}
    rvo, _, _ = visual_only(recs, "vo")
    out["stats"]["gru6d_vs_orbvo_rot"] = session_stats(rvo - rg, sess, names)
    out["stats"]["gru6d_vs_gyro_rot"] = session_stats(ry - rg, sess, names)
    for name in ("vo_gated", "slam_blend", "hf_gated"):
        out["stats"][f"{name}_vs_gru6d_rot"] = session_stats(rg - res[(name, "ref")][0], sess, names)
        for tmode in ("inertial", "ref"):
            out["stats"][f"{name}_vs_gru6d_trans_{tmode}"] = session_stats(tg - res[(name, tmode)][1], sess, names)
    slam_sessions = sorted({r["session"] for r in recs if r["slam_ok"]})
    in4 = np.array([r["session"] in slam_sessions for r in recs])
    m_slam = np.array([r["slam_ok"] for r in recs])
    m_vo4 = in4 & np.array([r["vo_ok"] for r in recs])
    rs, _, _ = visual_only(recs, "slam")
    out["subset"] = {"sessions": slam_sessions,
                     "on_slam_windows": {"N": int(m_slam.sum()), "orbslam3": float(np.nanmean(rs[m_slam])),
                                         "orbvo": float(np.nanmean(rvo[m_slam])), "gru6d": float(rg[m_slam].mean()), "gyro": float(ry[m_slam].mean())},
                     "on_orbvo_windows": {"N": int(m_vo4.sum()), "orbvo": float(np.nanmean(rvo[m_vo4])), "gru6d": float(rg[m_vo4].mean()),
                                          "gyro": float(ry[m_vo4].mean()), "hit_orbvo": float(np.mean(rvo[m_vo4] <= 5)),
                                          "hit_gru6d": float(np.mean(rg[m_vo4] <= 5)), "hit_gyro": float(np.mean(ry[m_vo4] <= 5))}}
    for tau in (2.0, 3.0, 5.0, 10.0):
        for gam in (0.1, 0.3, 0.5, 0.7):
            r, t, u = stream(recs, s0, "gated", "vo", tau=tau, gamma=gam, tmode="ref")
            out["sensitivity"].append({"tau": tau, "gamma": gam, "n_used": u, "rot_mean": float(r.mean()), "rot_p50": float(np.median(r)),
                                       "trans_mean": float(t.mean()), "hit_rot5": float(np.mean(r <= 5)), "hit_trans20": float(np.mean(t <= 20)),
                                       "max_excess": float((r - rg).max())})
    for s in seeds:
        scs = {k: inertial_scales(recs, k, s) for k in VIS}
        g = stream(recs, s, "gru")
        v = stream(recs, s, "gated", "vo", tmode="inertial", scales=scs["vo"])
        h = stream(recs, s, "gated", "hf", tmode="inertial", scales=scs["hf"])
        out["seed_range"][str(s)] = {"gru6d": [float(g[0].mean()), float(g[1].mean())], "vo_gated": [float(v[0].mean()), float(v[1].mean())],
                                     "hf_gated": [float(h[0].mean()), float(h[1].mean())]}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["rows"].items():
        print(f"{k:26s} rot {v['rot_mean']:.3f}/{v['rot_p50']:.3f}/{v['rot_p95']:.3f} trans {v['trans_mean']:.3f} "
              f"hit {v['hit_rot5']:.3f}/{v['hit_trans20']:.3f} N {v['N']}" + (f" used {v['n_visual_used']}" if "n_visual_used" in v else ""))
    for k, v in out["stats"].items():
        print(f"{k:34s} gain {v['gain']:+.3f} CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] better {v['sessions_better']}/11 p {v['p']:.4f}")
    print("bound:", out["bound"])
    print("seed range:", out["seed_range"])


if __name__ == "__main__":
    main()
