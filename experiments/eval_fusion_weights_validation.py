#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Variance-optimal rotation and translation blend weights from the validation sessions, applied to the test sessions.

On the 13 validation sessions, GRU6D (seed 0) and the ORB-VO and height-field increments are paired as on the test
sessions; on the increments that pass the consistency gate (tau = 3 deg), gamma*_R = MSE_imu / (MSE_imu + MSE_vis) of
the geodesic rotation errors and gamma*_t the same ratio of the squared translation errors with the inertial scale
(Proposition 3(ii)). The weights are then applied unchanged to the 11 test sessions.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from imu_camera_mlp import IMUCameraMLPTrainer, IMUCameraPoseGRU6D  # noqa: E402
from eval_vio_trajectory_rpe_ate import _interp_pose, _load_gt_poses_from_frames_labeled, _rel_end_frame  # noqa: E402
from eval_best_model_metrics import _project_to_so3_np  # noqa: E402

import eval_fixedtest_aligned as ev  # noqa: E402

SESS = IMUCAM_ROOT / "Data_IMU_Camera_Pose_Sequences"
ADS = ev.DS + "_offset-0.50"
TAU = 3.0
OUT = HERE / "fusion_weights_validation_results.json"


def gru6d_predict(names, seed=0):
    ref = IMUCameraMLPTrainer.from_checkpoint(str(ev.RUN / "best_model.pth"), device="cpu")
    m = IMUCameraPoseGRU6D(input_dim=21, embed_dim=256, gru_hidden_dim=512, gru_num_layers=2, head_dim=512, head_blocks=2, dropout=0.25)
    m.load_state_dict(torch.load(ev.PRED / f"gru6d_s{seed}.pt", map_location="cpu"))
    m.eval()
    out = {}
    for n in names:
        X = np.load(SESS / n / ADS / "X.npy")
        Xs = ref.scaler_imu.transform(X.reshape(-1, X.shape[-1])).astype(np.float32).reshape(X.shape)
        with torch.no_grad():
            out[n] = ref.scaler_pose.inverse_transform(m(torch.from_numpy(Xs)).numpy().astype(np.float64))
    return out


def records(names, pred):
    recs = []
    for n in names:
        sd = SESS / n
        y = np.load(sd / ev.DS / "y.npy")
        meta = list(csv.DictReader(open(sd / ev.DS / "meta.csv", encoding="utf-8")))
        gt_seq = _load_gt_poses_from_frames_labeled(str(sd / "frames_labeled.csv"), max_reproj_rmse=2.0,
                                                    chessboard_mode="majority", chessboard_name="", pos_scale=1.0)
        vis = {k: ev.load_vis(ev.VIS[k] / n / "traj_tum.txt", gt_seq) for k in ("vo", "hf")}
        t0 = float(gt_seq.t[0])
        for i, m in enumerate(meta):
            te = float(m["t_end"])
            ts = te - 0.5
            r = {"session": n, "R_gt": _project_to_so3_np(y[i, :9]), "t_gt": y[i, 9:12].astype(np.float64),
                 "gru": {0: (_project_to_so3_np(pred[n][i, :9]), pred[n][i, 9:12])}}
            ok = ts >= t0 and _interp_pose(gt_seq, te, max_gap_s=0.2) is not None and _interp_pose(gt_seq, ts, max_gap_s=0.2) is not None
            for k, (seq, s) in vis.items():
                r[k + "_ok"] = False
                if seq is None or not ok:
                    continue
                e, b = _interp_pose(seq, te, max_gap_s=0.2), _interp_pose(seq, ts, max_gap_s=0.2)
                if e is not None and b is not None:
                    R, tv = _rel_end_frame(e[0], e[1], b[0], b[1])
                    r.update({k + "_ok": True, "R_" + k: R, "t_" + k: tv.reshape(3), "s_" + k: s})
            recs.append(r)
    return recs


def gated(recs, key):
    return [r for r in recs if r[key + "_ok"] and ev.geo(r["R_" + key], r["gru"][0][0]) <= TAU]


def gamma_star(recs, key):
    sc = ev.inertial_scales(recs, key, 0)
    G = gated(recs, key)
    e_imu_R = np.array([ev.geo(r["gru"][0][0], r["R_gt"]) for r in G])
    e_vis_R = np.array([ev.geo(r["R_" + key], r["R_gt"]) for r in G])
    e_imu_t = np.array([np.linalg.norm(r["gru"][0][1] - r["t_gt"]) for r in G])
    e_vis_t = np.array([np.linalg.norm(r["t_" + key] * sc[r["session"]] - r["t_gt"]) for r in G])
    gR = float(np.mean(e_imu_R ** 2) / (np.mean(e_imu_R ** 2) + np.mean(e_vis_R ** 2)))
    gt = float(np.mean(e_imu_t ** 2) / (np.mean(e_imu_t ** 2) + np.mean(e_vis_t ** 2)))
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    best_R = min(grid, key=lambda g: blend(recs, key, g, 0.0, sc)[0].mean())
    best_t = min(grid, key=lambda g: blend(recs, key, 0.0, g, sc)[1].mean())
    return {"n_gated": len(G), "n_windows": len(recs), "rmse_imu_R": float(np.sqrt(np.mean(e_imu_R ** 2))),
            "rmse_vis_R": float(np.sqrt(np.mean(e_vis_R ** 2))), "rmse_imu_t": float(np.sqrt(np.mean(e_imu_t ** 2))),
            "rmse_vis_t": float(np.sqrt(np.mean(e_vis_t ** 2))), "gamma_R": gR, "gamma_t": gt,
            "grid_best_gamma_R": float(best_R), "grid_best_gamma_t": float(best_t)}


def blend(recs, key, gR, gt, scales, tmode="inertial"):
    rot, tr = [], []
    for r in recs:
        R, t = r["gru"][0]
        if r[key + "_ok"] and ev.geo(r["R_" + key], R) <= TAU:
            R = ev.slerp(R, r["R_" + key], gR) if gR > 0 else R
            s = r["s_" + key] if tmode == "ref" else scales[r["session"]]
            t = gt * r["t_" + key] * s + (1 - gt) * t
        rot.append(ev.geo(R, r["R_gt"]))
        tr.append(float(np.linalg.norm(t - r["t_gt"])))
    return np.asarray(rot), np.asarray(tr)


def main():
    val = json.loads((HERE / "validation_sessions.json").read_text(encoding="utf-8"))["validation"]
    vrecs = records(val, gru6d_predict(val))
    out = {"protocol": __doc__.strip().splitlines()[0], "tau": TAU, "validation_sessions": val, "weights": {}, "test": {}}
    trecs, _ = ev.build_records()
    sess = np.array([r["session"] for r in trecs])
    names = list(dict.fromkeys(sess.tolist()))
    rg = np.array([ev.geo(r["gru"][0][0], r["R_gt"]) for r in trecs])
    tg = np.array([np.linalg.norm(r["gru"][0][1] - r["t_gt"]) for r in trecs])
    for key in ("vo", "hf"):
        w = gamma_star(vrecs, key)
        out["weights"][key] = w
        sc = ev.inertial_scales(trecs, key, 0)
        res = {}
        for tmode in ("inertial", "ref"):
            r, t = blend(trecs, key, w["gamma_R"], w["gamma_t"], sc, tmode)
            res[tmode] = {"rot_mean": float(r.mean()), "rot_p50": float(np.median(r)), "rot_p95": float(np.percentile(r, 95)),
                          "trans_mean": float(t.mean()), "trans_p50": float(np.median(t)), "trans_p95": float(np.percentile(t, 95)),
                          "hit_rot5": float(np.mean(r <= 5)), "hit_trans20": float(np.mean(t <= 20)),
                          "max_excess_deg": float((r - rg).max()), "bound_deg": w["gamma_R"] * TAU,
                          "rot_vs_gru6d": ev.session_stats(rg - r, sess, names), "trans_vs_gru6d": ev.session_stats(tg - t, sess, names)}
        out["test"][key] = res
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key in ("vo", "hf"):
        w = out["weights"][key]
        print(f"{key}: validation gated {w['n_gated']}/{w['n_windows']}  gamma*_R {w['gamma_R']:.3f} (grid {w['grid_best_gamma_R']})  "
              f"gamma*_t {w['gamma_t']:.3f} (grid {w['grid_best_gamma_t']})  rmse imu/vis R {w['rmse_imu_R']:.2f}/{w['rmse_vis_R']:.2f} "
              f"t {w['rmse_imu_t']:.2f}/{w['rmse_vis_t']:.2f}")
        for tmode, v in out["test"][key].items():
            a, b = v["rot_vs_gru6d"], v["trans_vs_gru6d"]
            print(f"   test {tmode:8s} rot {v['rot_mean']:.3f} ({a['gain']:+.3f}, {a['sessions_better']}/11, p {a['p']:.3f}) "
                  f"trans {v['trans_mean']:.3f} ({b['gain']:+.3f}, {b['sessions_better']}/11, p {b['p']:.3f}) "
                  f"max_excess {v['max_excess_deg']:.2f} <= {v['bound_deg']:.2f}")


if __name__ == "__main__":
    main()
