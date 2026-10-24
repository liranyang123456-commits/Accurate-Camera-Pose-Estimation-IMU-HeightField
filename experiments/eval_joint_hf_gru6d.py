#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full pipeline on the 11 held-out sessions: height-field tracker + GRU6D + consistency gate.

Window pool, reference interpolation, relative-pose convention, gate (tau_rot = 3 deg) and blend weight
(gamma = 0.5) are those of the ORB-VO experiment (eval_combined_vi_and_stats.py). Pre-specified primary
configuration for the height-field front end: blend rotation only and keep the GRU6D translation, because
the height-field transform is a shape alignment in (u, v, z) whose translation is not a camera translation.
Secondary: full SE(3) blend with the inertial scale. References: zero rotation, ORB-VO blend.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.spatial.transform import Rotation, Slerp

from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from eval_vio_fixedtest_aggregate import _load_fixed_test_sessions_from_train_config  # noqa: E402
from eval_vio_trajectory_rpe_ate import (  # noqa: E402
    PoseSeq, _interp_pose, _load_gt_poses_from_frames_labeled, _load_traj_tum, _rel_end_frame)
from run_classical_vo_vio_fixedtest import _estimate_scale_from_gt  # noqa: E402
from imu_camera_mlp import IMUCameraMLPTrainer  # noqa: E402
from eval_best_model_metrics import _project_to_so3_np  # noqa: E402

ROOT = IMUCAM_ROOT
RUN = ROOT / "out_train_fixed_seq_20260310_031242"
VO_ROOT = ROOT / "vio_runs_classical" / "orb_vo"
HF_ROOT = Path(__file__).resolve().parent / "hf_sessions"
DS = "fixed_seq_len60_win0.500s_delta_window"
TAU, GAMMA = 3.0, 0.5


def geo(R1, R2):
    return float(np.degrees(np.arccos(np.clip((float(np.sum(R1 * R2)) - 1.0) * 0.5, -1.0, 1.0))))


def slerp(Ra, Rb, w):
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb], 0)))([w]).as_matrix()[0]


def load_vis(path, gt_seq, scale_from_gt):
    if not path.exists() or path.stat().st_size == 0:
        return None, 1.0
    est = _load_traj_tum(str(path), time_unit="s", pos_scale=1.0, pose_mode="T_WC")
    s = float(_estimate_scale_from_gt(est, gt_seq, window_s=0.5)) if scale_from_gt else 1.0
    return PoseSeq(t=est.t, p=est.p * s, q=est.q), s


def build_records():
    sessions = _load_fixed_test_sessions_from_train_config(str(RUN / "train_config.json"))
    tr = IMUCameraMLPTrainer.from_checkpoint(str(RUN / "best_model.pth"), device="cpu")
    tr.model.eval()
    recs = []
    for sd in sessions:
        sd = Path(sd)
        X = np.load(sd / DS / "X.npy")
        y = np.load(sd / DS / "y.npy")
        meta = list(csv.DictReader(open(sd / DS / "meta.csv", encoding="utf-8")))
        Xs = tr.scaler_imu.transform(X.reshape(-1, X.shape[-1])).astype(np.float32).reshape(X.shape)
        with torch.no_grad():
            pred = tr.scaler_pose.inverse_transform(tr.model(torch.from_numpy(Xs)).numpy())
        gt_seq = _load_gt_poses_from_frames_labeled(str(sd / "frames_labeled.csv"), max_reproj_rmse=2.0,
                                                    chessboard_mode="majority", chessboard_name="", pos_scale=1.0)
        t0 = float(gt_seq.t[0])
        vo, s_vo = load_vis(VO_ROOT / sd.name / "traj_tum.txt", gt_seq, True)
        hf, _ = load_vis(HF_ROOT / sd.name / "traj_tum.txt", gt_seq, False)
        _, s_hf_ref = load_vis(HF_ROOT / sd.name / "traj_tum.txt", gt_seq, True)
        for k, m in enumerate(meta):
            t_end = float(m.get("t_end", m.get("frame_timestamp", 0.0)))
            t_start = t_end - 0.5
            r = {"session": sd.name, "R_gt": _project_to_so3_np(y[k, :9]), "t_gt": y[k, 9:12].astype(np.float64),
                 "R_gru": _project_to_so3_np(pred[k, :9]), "t_gru": pred[k, 9:12].astype(np.float64),
                 "vo_ok": False, "hf_ok": False, "s_vo": s_vo, "s_hf_ref": s_hf_ref}
            if t_start >= t0 and _interp_pose(gt_seq, t_end, max_gap_s=0.2) is not None and _interp_pose(gt_seq, t_start, max_gap_s=0.2) is not None:
                for name, seq in (("vo", vo), ("hf", hf)):
                    if seq is None:
                        continue
                    e, s = _interp_pose(seq, t_end, max_gap_s=0.2), _interp_pose(seq, t_start, max_gap_s=0.2)
                    if e is not None and s is not None:
                        R, t = _rel_end_frame(e[0], e[1], s[0], s[1])
                        r[name + "_ok"], r["R_" + name], r["t_" + name] = True, R, t.reshape(3)
            recs.append(r)
    return recs


def inertial_scales(recs, key):
    out = {}
    for s in sorted({r["session"] for r in recs}):
        div = (lambda r: r["s_vo"]) if key == "vo" else (lambda r: 1.0)
        ratios = [np.linalg.norm(r["t_gru"]) / np.linalg.norm(r["t_" + key] / div(r)) for r in recs
                  if r["session"] == s and r[key + "_ok"] and np.linalg.norm(r["t_" + key]) > 1e-9]
        out[s] = float(np.median(ratios)) if ratios else 1.0
    return out


def stream(recs, mode, scales=None):
    rot, tr, n_used = [], [], 0
    for r in recs:
        R, t = r["R_gru"], r["t_gru"]
        if mode == "zero_rot":
            R = np.eye(3)
        elif mode.startswith("hf") and r["hf_ok"]:
            if mode in ("hf_only", "hf_only_ref"):
                R = r["R_hf"]
                n_used += 1
                if mode == "hf_only_ref":
                    t = r["t_hf"] * r["s_hf_ref"]
            elif geo(r["R_hf"], r["R_gru"]) <= TAU:
                n_used += 1
                R = slerp(r["R_gru"], r["R_hf"], GAMMA)
                if mode == "hf_gated_se3":
                    t = GAMMA * r["t_hf"] * scales[r["session"]] + (1 - GAMMA) * r["t_gru"]
                elif mode == "hf_gated_se3_ref":
                    t = GAMMA * r["t_hf"] * r["s_hf_ref"] + (1 - GAMMA) * r["t_gru"]
        elif mode == "vo_gated" and r["vo_ok"] and geo(r["R_vo"], r["R_gru"]) <= TAU:
            n_used += 1
            R = slerp(r["R_gru"], r["R_vo"], GAMMA)
            t = GAMMA * (r["t_vo"] / r["s_vo"]) * scales[r["session"]] + (1 - GAMMA) * r["t_gru"]
        elif mode == "vo_gated_rot" and r["vo_ok"] and geo(r["R_vo"], r["R_gru"]) <= TAU:
            n_used += 1
            R = slerp(r["R_gru"], r["R_vo"], GAMMA)
        rot.append(geo(R, r["R_gt"]))
        tr.append(float(np.linalg.norm(t - r["t_gt"])))
    return np.asarray(rot), np.asarray(tr), n_used


def session_stats(diff, sess, names, B=5000, seed=7):
    groups = [diff[sess == s] for s in names]
    per = np.array([g.mean() for g in groups])
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(B)]
    return {"window_mean_gain": float(diff.mean()), "cluster_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "sessions_improved": int((per > 0).sum()), "sessions_worse": int((per < 0).sum()), "n_sessions": len(names),
            "session_wilcoxon_p": float(stats.wilcoxon(per).pvalue) if np.any(per != 0) else 1.0,
            "per_session_gain": [round(float(x), 4) for x in per]}


def main():
    recs = build_records()
    sess = np.array([r["session"] for r in recs])
    names = list(dict.fromkeys(sess.tolist()))
    sc_hf, sc_vo = inertial_scales(recs, "hf"), inertial_scales(recs, "vo")
    r_g, t_g, _ = stream(recs, "gru")
    out = {"protocol": __doc__.strip().splitlines()[0], "tau_rot_deg": TAU, "gamma": GAMMA, "N": len(recs),
           "windows_with_hf_support": int(sum(r["hf_ok"] for r in recs)), "windows_with_vo_support": int(sum(r["vo_ok"] for r in recs)),
           "streams": {}}
    hf_mask = np.array([r["hf_ok"] for r in recs])
    for mode, scales in (("zero_rot", None), ("hf_only", None), ("hf_only_ref", None), ("hf_gated_rot", None), ("hf_gated_se3", sc_hf),
                         ("hf_gated_se3_ref", None), ("vo_gated", sc_vo), ("vo_gated_rot", None)):
        r, t, n = stream(recs, mode, scales)
        if mode == "hf_only_ref":
            r, t = r[hf_mask], t[hf_mask]
        s = {"rot_mean": float(r.mean()), "rot_p50": float(np.median(r)), "rot_p95": float(np.percentile(r, 95)),
             "trans_mean": float(t.mean()), "trans_p50": float(np.median(t)), "trans_p95": float(np.percentile(t, 95)),
             "hit_rot5": float(np.mean(r <= 5.0)), "hit_trans20": float(np.mean(t <= 20.0)), "N": int(len(r)), "n_visual_used": int(n),
             }
        if mode != "hf_only_ref":
            s.update({"rot_gain_vs_gru6d": session_stats(r_g - r, sess, names), "trans_gain_vs_gru6d": session_stats(t_g - t, sess, names)})
        if mode == "hf_only":
            s["rot_mean_on_supported"] = float(r[hf_mask].mean())
            s["gru6d_rot_mean_on_supported"] = float(r_g[hf_mask].mean())
            s["zero_rot_mean_on_supported"] = float(np.mean([geo(np.eye(3), rr["R_gt"]) for rr in np.array(recs, dtype=object)[hf_mask]]))
        out["streams"][mode] = s
    out["streams"]["gru6d"] = {"rot_mean": float(r_g.mean()), "rot_p50": float(np.median(r_g)), "trans_mean": float(t_g.mean()), "hit_rot5": float(np.mean(r_g <= 5.0))}
    p = Path(__file__).resolve().parent / "joint_hf_gru6d_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out["streams"].items():
        line = f"{k:14s} rot {v['rot_mean']:.3f} (p50 {v['rot_p50']:.3f}) trans {v['trans_mean']:.3f}"
        if "rot_gain_vs_gru6d" in v:
            g = v["rot_gain_vs_gru6d"]
            line += f" | used {v['n_visual_used']} | rot gain {g['window_mean_gain']:+.3f} CI {[round(x, 3) for x in g['cluster_ci95']]} sessions +{g['sessions_improved']}/-{g['sessions_worse']} p {g['session_wilcoxon_p']:.4f}"
        print(line)
    print("hf_only on supported windows:", {k: out["streams"]["hf_only"][k] for k in ("rot_mean_on_supported", "gru6d_rot_mean_on_supported", "zero_rot_mean_on_supported")})


if __name__ == "__main__":
    main()
