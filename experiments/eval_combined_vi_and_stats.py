#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Visual-only vs. IMU-only vs. Support-Gated Combined Visual-Inertial
on the exact 11 fixed-test sessions (N = 2312 windows, with exact N = 419 ORB-SLAM3
windows and N = 2296 ORB-VO windows matching Manuscript Tables 2-4), plus:
  - 95% bootstrap confidence intervals (5000 resamples)
  - Two-sided paired Wilcoxon signed-rank p-values
  - Sensitivity grid over consistency gate threshold tau_rot and blending weight gamma
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, r"D:\reloc3r")
from eval_vio_fixedtest_aggregate import _load_fixed_test_sessions_from_train_config
from eval_vio_trajectory_rpe_ate import (
    PoseSeq,
    _interp_pose,
    _load_gt_poses_from_frames_labeled,
    _load_traj_tum,
    _rel_end_frame,
)
from run_classical_vo_vio_fixedtest import _estimate_scale_from_gt
from imu_camera_mlp import IMUCameraMLPTrainer
from eval_best_model_metrics import _project_to_so3_np

ROOT = Path(r"D:\reloc3r")
TRAIN_CFG = ROOT / r"out_train_fixed_seq_20260310_031242\train_config.json"
CKPT_PATH = ROOT / r"out_train_fixed_seq_20260310_031242\best_model.pth"
SLAM_ROOT = ROOT / r"vio_runs\ORB-SLAM3-mono"
VO_ROOT = ROOT / r"vio_runs_classical\orb_vo"


def bootstrap_ci(arr: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def so3_geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (float(np.sum(R1 * R2)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> None:
    sessions = _load_fixed_test_sessions_from_train_config(str(TRAIN_CFG))
    trainer = IMUCameraMLPTrainer.from_checkpoint(str(CKPT_PATH), device="cpu")
    trainer.model.eval()

    records = []

    for sd_str in sessions:
        sd = Path(sd_str)
        sname = sd.name
        ds_dir = sd / "fixed_seq_len60_win0.500s_delta_window"
        X = np.load(ds_dir / "X.npy")
        y = np.load(ds_dir / "y.npy")
        with open(ds_dir / "meta.csv", "r", encoding="utf-8") as f:
            meta_rows = list(csv.DictReader(f))

        Xs = trainer.scaler_imu.transform(X.reshape(-1, X.shape[-1])).astype(np.float32).reshape(X.shape)
        with torch.no_grad():
            pred_s = trainer.model(torch.from_numpy(Xs)).numpy()
        pred_u = trainer.scaler_pose.inverse_transform(pred_s)

        gt_csv = sd / "frames_labeled.csv"
        gt_seq = _load_gt_poses_from_frames_labeled(
            str(gt_csv), max_reproj_rmse=2.0, chessboard_mode="majority", chessboard_name="", pos_scale=1.0
        )
        t_gt0 = float(gt_seq.t[0])

        slam_seq = None
        slam_traj = SLAM_ROOT / sname / "traj_tum.txt"
        if slam_traj.exists() and slam_traj.stat().st_size > 0:
            try:
                est_s = _load_traj_tum(str(slam_traj), time_unit="s", pos_scale=1.0, pose_mode="T_WC")
                scale_s = _estimate_scale_from_gt(est_s, gt_seq, window_s=0.5)
                slam_seq = PoseSeq(t=est_s.t, p=est_s.p * float(scale_s), q=est_s.q)
            except Exception:
                slam_seq = None

        vo_seq = None
        vo_traj = VO_ROOT / sname / "traj_tum.txt"
        if vo_traj.exists() and vo_traj.stat().st_size > 0:
            try:
                est_v = _load_traj_tum(str(vo_traj), time_unit="s", pos_scale=1.0, pose_mode="T_WC")
                scale_v = _estimate_scale_from_gt(est_v, gt_seq, window_s=0.5)
                vo_seq = PoseSeq(t=est_v.t, p=est_v.p * float(scale_v), q=est_v.q)
            except Exception:
                vo_seq = None

        for k, mrow in enumerate(meta_rows):
            t_end = float(mrow.get("t_end", mrow.get("frame_timestamp", 0.0)))
            t_start = float(t_end - 0.5)

            R_gt = _project_to_so3_np(y[k, :9])
            t_gt = y[k, 9:12].astype(np.float64)

            R_gru = _project_to_so3_np(pred_u[k, :9])
            t_gru = pred_u[k, 9:12].astype(np.float64)

            rec = {
                "R_gt": R_gt,
                "t_gt": t_gt,
                "R_gru": R_gru,
                "t_gru": t_gru,
                "slam_ok": False,
                "R_slam": None,
                "t_slam": None,
                "vo_ok": False,
                "R_vo": None,
                "t_vo": None,
            }

            if t_start >= t_gt0:
                gt_e = _interp_pose(gt_seq, t_end, max_gap_s=0.2)
                gt_s = _interp_pose(gt_seq, t_start, max_gap_s=0.2)
                if gt_e is not None and gt_s is not None:
                    if slam_seq is not None:
                        se = _interp_pose(slam_seq, t_end, max_gap_s=0.2)
                        ss = _interp_pose(slam_seq, t_start, max_gap_s=0.2)
                        if se is not None and ss is not None:
                            R_slam, t_slam = _rel_end_frame(se[0], se[1], ss[0], ss[1])
                            rec["slam_ok"] = True
                            rec["R_slam"] = R_slam
                            rec["t_slam"] = t_slam.reshape(3)
                    if vo_seq is not None:
                        ve = _interp_pose(vo_seq, t_end, max_gap_s=0.2)
                        vs = _interp_pose(vo_seq, t_start, max_gap_s=0.2)
                        if ve is not None and vs is not None:
                            R_vo, t_vo = _rel_end_frame(ve[0], ve[1], vs[0], vs[1])
                            rec["vo_ok"] = True
                            rec["R_vo"] = R_vo
                            rec["t_vo"] = t_vo.reshape(3)

            records.append(rec)

    def eval_stream(mode: str, tau_rot: float = 3.0, gamma: float = 0.5):
        rots, trans = [], []
        n_vis_used = 0
        for r in records:
            R_gt, t_gt = r["R_gt"], r["t_gt"]
            R_gru, t_gru = r["R_gru"], r["t_gru"]
            if mode == "gru":
                R_est, t_est = R_gru, t_gru
            elif mode == "slam_blend":
                if r["slam_ok"]:
                    n_vis_used += 1
                    if gamma >= 1.0:
                        R_est, t_est = r["R_slam"], r["t_slam"]
                    else:
                        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R_gru, r["R_slam"]], 0)))
                        R_est = slerp([gamma]).as_matrix()[0]
                        t_est = gamma * r["t_slam"] + (1.0 - gamma) * t_gru
                else:
                    R_est, t_est = R_gru, t_gru
            elif mode == "vo_gated":
                if r["vo_ok"]:
                    disagree = so3_geodesic_deg(r["R_vo"], R_gru)
                    if disagree <= tau_rot:
                        n_vis_used += 1
                        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R_gru, r["R_vo"]], 0)))
                        R_est = slerp([gamma]).as_matrix()[0]
                        t_est = gamma * r["t_vo"] + (1.0 - gamma) * t_gru
                    else:
                        R_est, t_est = R_gru, t_gru
                else:
                    R_est, t_est = R_gru, t_gru
            else:
                raise ValueError(mode)
            rots.append(so3_geodesic_deg(R_est, R_gt))
            trans.append(float(np.linalg.norm(t_est - t_gt)))
        return np.asarray(rots), np.asarray(trans), n_vis_used

    def summarize(name: str, r: np.ndarray, t: np.ndarray, n_vis: int):
        return {
            "name": name,
            "N": int(len(r)),
            "n_vis_used": int(n_vis),
            "rot_mean": float(r.mean()),
            "rot_p50": float(np.percentile(r, 50)),
            "rot_p95": float(np.percentile(r, 95)),
            "rot_ci95": list(bootstrap_ci(r)),
            "trans_mean": float(t.mean()),
            "trans_p50": float(np.percentile(t, 50)),
            "trans_p95": float(np.percentile(t, 95)),
            "trans_ci95": list(bootstrap_ci(t)),
            "hit_rot5": float(np.mean(r <= 5.0)),
            "hit_trans20": float(np.mean(t <= 20.0)),
        }

    r_gru, t_gru, _ = eval_stream("gru")
    r_s_sw, t_s_sw, n_slam = eval_stream("slam_blend", gamma=1.0)
    r_s_bl, t_s_bl, _ = eval_stream("slam_blend", gamma=0.7)
    r_v_gt, t_v_gt, n_vo = eval_stream("vo_gated", tau_rot=3.0, gamma=0.5)

    # Sensitivity grid for Consistency-Gated ORB-VO + GRU6D
    sens_grid = []
    for tau in [2.0, 3.0, 5.0, 10.0]:
        for gam in [0.3, 0.5, 0.7]:
            rv, tv, nv = eval_stream("vo_gated", tau_rot=tau, gamma=gam)
            sens_grid.append({
                "tau_rot_deg": tau,
                "gamma": gam,
                "n_gated_in": int(nv),
                "rot_mean": round(float(rv.mean()), 3),
                "rot_p50": round(float(np.percentile(rv, 50)), 3),
                "trans_mean": round(float(tv.mean()), 3),
                "hit_rot5": round(float(np.mean(rv <= 5.0)), 3),
                "hit_trans20": round(float(np.mean(tv <= 20.0)), 3),
            })

    out = {
        "gru6d": summarize("GRU6D (IMU-only)", r_gru, t_gru, 0),
        "slam_switch": summarize("Support-Gated ORB-SLAM3 + GRU6D (gamma=1.0)", r_s_sw, t_s_sw, n_slam),
        "slam_blend": summarize("Support-Blended ORB-SLAM3 + GRU6D (gamma=0.7)", r_s_bl, t_s_bl, n_slam),
        "vo_gated": summarize("Consistency-Gated ORB-VO + GRU6D (tau=3, gamma=0.5)", r_v_gt, t_v_gt, n_vo),
        "wilcoxon_p": {
            "slam_blend_vs_gru_rot": float(stats.wilcoxon(r_s_bl[r_s_bl != r_gru], r_gru[r_s_bl != r_gru]).pvalue),
            "slam_blend_vs_gru_trans": float(stats.wilcoxon(t_s_bl[t_s_bl != t_gru], t_gru[t_s_bl != t_gru]).pvalue),
            "vo_gated_vs_gru_rot": float(stats.wilcoxon(r_v_gt[r_v_gt != r_gru], r_gru[r_v_gt != r_gru]).pvalue),
            "vo_gated_vs_gru_trans": float(stats.wilcoxon(t_v_gt[t_v_gt != t_gru], t_gru[t_v_gt != t_gru]).pvalue),
        },
        "sensitivity_grid": sens_grid,
    }

    out_path = Path(__file__).resolve().parent / "combined_vi_and_stats.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
