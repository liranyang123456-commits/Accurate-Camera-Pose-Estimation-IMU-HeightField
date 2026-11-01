#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-level comparison with working tightly coupled VIO on TUM-VI room1 (hardware-synchronized, Kalibr-calibrated).

Windows: 0.5 s ending at every camera frame of room1 (build_tumvi_windows.py; motion-capture reference, camera frame).
Inertial increment: gyroscope integration with the Kalibr R_CI and the translation of the regressors trained on rooms
2-5 (validation room6; train_miti_gru6d.py --windows tumvi_windows.npz). Visual front ends (run_visual_tumvi.py) at 20 Hz
and 6.7 Hz with the reference scale (visual only) or the inertial scale (fused); blend weights selected on the
checkerboard validation sessions of the primary rig and applied unchanged. VIO: official ORB-SLAM3 mono-inertial and
VINS-Fusion stereo-inertial at 20, 10, and 6.7 Hz (run_tumvi_framerate.py) and ORB-SLAM3 under the controlled degradations
of run_tumvi_vio_degradation.py; their metric trajectories are converted to the camera frame with the extrinsics each run
assumed and scored on the windows whose endpoints they cover. Writes tumvi_fusion_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from eval_vio_trajectory_rpe_ate import _interp_pose, _load_traj_tum, _rel_end_frame  # noqa: E402

ROOT = IMUCAM_ROOT
VIS = ROOT / "tumvi_visual"
FR = ROOT / "tumvi_room1" / "framerate"
DEG = ROOT / "tumvi_room1" / "degradation"
PRED = HERE / "tumvi_regressors"
OUT = HERE / "tumvi_fusion_results.json"
TAU = 3.0
GAP = 0.2
W_RIG = {"hf": (0.05, 0.40), "vo": (0.05, 0.55), "hvo": (0.05, 0.85)}
DIRS = {"hf": "hf", "vo": "orb_vo", "hvo": "orb_hvo"}
RATES = {1: "20hz", 3: "6.7hz"}
NONOVL = 10


def geo(A, B):
    return np.degrees(Rotation.from_matrix(np.transpose(A, (0, 2, 1)) @ B).magnitude())


def slerp1(Ra, Rb, w):
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))([w]).as_matrix()[0]


def summ(r, t):
    r, t = np.asarray(r, float), np.asarray(t, float)
    return {"rot_mean": float(r.mean()), "rot_p50": float(np.median(r)), "rot_p95": float(np.percentile(r, 95)),
            "trans_mean": float(t.mean()), "trans_p50": float(np.median(t)), "trans_p95": float(np.percentile(t, 95)), "N": int(r.size)}


def windows_from_traj(seq, t0, t1, T_bc=None):
    """Relative camera motion over each window from a world<-camera (or world<-body, with T_bc) trajectory."""
    N = len(t0)
    ok = np.zeros(N, bool)
    R = np.repeat(np.eye(3)[None], N, 0)
    t = np.zeros((N, 3))
    for i in range(N):
        e, s = _interp_pose(seq, t1[i], max_gap_s=GAP), _interp_pose(seq, t0[i], max_gap_s=GAP)
        if e is None or s is None:
            continue
        (Re, pe), (Rs, ps) = e, s
        if T_bc is not None:
            pe, Re = pe + Re @ T_bc[:3, 3], Re @ T_bc[:3, :3]
            ps, Rs = ps + Rs @ T_bc[:3, 3], Rs @ T_bc[:3, :3]
        Rr, tr = _rel_end_frame(Re, pe, Rs, ps)
        ok[i], R[i], t[i] = True, Rr, tr.reshape(3)
    return ok, R, t


def main():
    z = np.load(HERE / "tumvi_windows.npz", allow_pickle=True)
    te = z["split"].astype(str) == "test"
    t0, t1 = z["t_start"][te], z["t_end"][te]
    y = z["y"][te].astype(np.float64)
    R_gt, t_gt = y[:, :9].reshape(-1, 3, 3), y[:, 9:12]
    R_gyro = z["R_gyro"][te]
    T_bc = z["T_b_c"].astype(np.float64)
    T_bc_mm = T_bc.copy()
    T_bc_mm[:3, 3] *= 1000.0
    N = len(t0)
    res = json.loads((PRED / "miti_gru6d_results.json").read_text(encoding="utf-8"))
    preds = {k: np.load(PRED / f"pred_{k}.npz")["room1"] for k in res["runs"]}
    for k, p in preds.items():
        assert len(p) == N, (k, len(p), N)

    def proj(p):
        U, _, Vt = np.linalg.svd(p[:, :9].reshape(-1, 3, 3))
        d = np.sign(np.linalg.det(U @ Vt))
        U[:, :, -1] *= d[:, None]
        return U @ Vt

    t_src = {k: p[:, 9:12].astype(np.float64) for k, p in preds.items()}
    main_key = "gru6d_s0"
    e_gyro = geo(R_gyro, R_gt)
    t_zero = np.linalg.norm(t_gt, axis=1)
    rows = {"zero_motion": (geo(np.repeat(np.eye(3)[None], N, 0), R_gt), t_zero, np.ones(N, bool)),
            "gyro_zeroT": (e_gyro, t_zero, np.ones(N, bool)),
            "gru6d_full": (geo(proj(preds[main_key]), R_gt), np.linalg.norm(t_src[main_key] - t_gt, axis=1), np.ones(N, bool))}
    for k in sorted(t_src):
        rows[f"inertial_{k}"] = (e_gyro, np.linalg.norm(t_src[k] - t_gt, axis=1), np.ones(N, bool))
    out = {"protocol": __doc__.strip().splitlines()[0], "N": N, "tau": TAU, "rows": {}, "rows_non_overlapping": {}, "coverage": {}, "damage": {},
           "encoders_test_trans_mm": {k: float(np.linalg.norm(v - t_gt, axis=1).mean()) for k, v in t_src.items()},
           "encoders_test_rot_deg": {k: float(geo(proj(p), R_gt).mean()) for k, p in preds.items()},
           "zero_motion": {"rot_deg": float(rows["zero_motion"][0].mean()), "trans_mm": float(t_zero.mean())}}

    # visual front ends and gated blends
    for key, d in DIRS.items():
        for k, rate in RATES.items():
            path = VIS / d / f"room1_s{k}" / "traj_tum.txt"
            if not path.exists():
                continue
            ok, Rv, tv = windows_from_traj(_load_traj_tum(str(path), time_unit="s", pos_scale=1.0, pose_mode="T_WC"), t0, t1)
            good = ok & (np.linalg.norm(tv, axis=1) > 1e-9)

            def scale(src):
                return float(np.median(np.linalg.norm(src[good], axis=1) / np.linalg.norm(tv[good], axis=1)))

            s_ref = scale(t_gt)
            tag = f"{key}_{rate}"
            gate = good & (geo(Rv, R_gyro) <= TAU)
            out["coverage"][tag] = {"supported": float(good.mean()), "gate_accept": float(gate.mean())}
            rows[f"{tag}_only"] = (geo(Rv, R_gt), np.linalg.norm(s_ref * tv - t_gt, axis=1), good)
            for src_name in ("gru6d_s0", "tlio_resnet_s0"):
                if src_name not in t_src:
                    continue
                tsrc = t_src[src_name]
                s_in = scale(tsrc)
                for name, (gR, gt, gated) in {"rig_weights": (*W_RIG[key], True), "default": (0.5, 0.5, True), "nogate": (0.5, 0.5, False),
                                             "gammaR0_ungated": (0.0, W_RIG[key][1], False)}.items():
                    R = R_gyro.copy()
                    t = tsrc.copy()
                    use = gate if gated else good
                    for i in np.where(use)[0]:
                        if gR > 0:
                            R[i] = slerp1(R_gyro[i], Rv[i], gR)
                        t[i] = gt * s_in * tv[i] + (1 - gt) * tsrc[i]
                    r_e, t_e = geo(R, R_gt), np.linalg.norm(t - t_gt, axis=1)
                    rk = f"{tag}_{name}" + ("" if src_name == "gru6d_s0" else "_tlio")
                    rows[rk] = (r_e, t_e, np.ones(N, bool))
                    base = np.linalg.norm(tsrc - t_gt, axis=1)
                    out["damage"][rk] = {"used": int(use.sum()), "max_rot_increase_deg": float((r_e - e_gyro).max()),
                                         "max_trans_increase_mm": float((t_e - base).max())}

    # tightly coupled VIO (metric, IMU-frame trajectories converted to the camera frame with the extrinsics each run assumed)
    vio = {}
    for k in (1, 2, 3):
        vio[f"orbslam3_vi_{20 // k if k < 3 else 6.7}hz".replace("6.7hz", "6.7hz")] = (FR / f"orb_s{k}" / f"f_tumvi_room1_s{k}.txt", "ns", T_bc_mm)
        vio[f"vins_fusion_{20 // k if k < 3 else 6.7}hz"] = (FR / f"vins_s{k}" / "traj_tum.txt", "s", T_bc_mm)
    deg = json.loads((HERE / "tumvi_vio_degradation_results.json").read_text(encoding="utf-8")) if (HERE / "tumvi_vio_degradation_results.json").exists() else {"runs": {}}
    out["degradation"] = {}
    for name, r in deg["runs"].items():
        T_assumed = T_bc_mm.copy()
        if r["lever_arm_zero"]:
            T_assumed[:3, 3] = 0.0
        vio[f"orbslam3_vi_{name}"] = (Path(r["trajectory"]), "ns", T_assumed)
        log = Path(r["trajectory"]).parent.with_suffix(".log")
        txt = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
        import re  # noqa: E402
        mm = re.findall(r"There are (\d+) maps in the atlas", txt)
        out["degradation"][name] = {k: r.get(k) for k in ("frame_rate_hz", "lever_arm_zero", "imu_offset_s", "num_est_poses", "ate_se3_rmse_m",
                                                         "rpe_rot_mean_deg", "rpe_trans_mean_m")}
        out["degradation"][name]["maps_in_atlas"] = int(mm[-1]) if mm else None
        out["degradation"][name]["imu_initialized_viba2"] = txt.count("end VIBA 2")
    # interpolation floor: the reference itself, kept only at the frames an estimator at 20/10/6.7 Hz processes
    gt_t, gt = (lambda a: (a[:, 0] * 1e-9, a[:, 1:]))(np.genfromtxt(ROOT / "tumvi_room1" / "extracted" / "dataset-room1_512" / "mav0"
                                                                    / "state_groundtruth_estimate0" / "data.csv", delimiter=",", skip_header=1))
    o = np.argsort(gt_t)
    gt_t, gt = gt_t[o], gt[o]
    sl_gt = Slerp(gt_t, Rotation.from_quat(gt[:, [4, 5, 6, 3]]))
    for k in (1, 2, 3):
        ts = np.loadtxt(FR / f"ts_stride{k}.txt", dtype=np.int64) * 1e-9
        ts = ts[(ts > gt_t[0]) & (ts < gt_t[-1])]
        p = np.stack([np.interp(ts, gt_t, gt[:, j]) for j in range(3)], 1)
        q = sl_gt(ts).as_quat()
        tmp = OUT.parent / f"_ref_stride{k}.txt"
        np.savetxt(tmp, np.column_stack([ts, p, q]), fmt="%.9f")
        vio[f"reference_subsampled_{({1: '20', 2: '10', 3: '6.7'})[k]}hz"] = (tmp, "s", T_bc_mm)
    out["vio"] = {}
    for name, (path, unit, T_assumed) in vio.items():
        if not path.exists() or path.stat().st_size == 0:
            out["vio"][name] = {"covered": 0.0, "note": "no trajectory"}
            continue
        ok, Rv, tv = windows_from_traj(_load_traj_tum(str(path), time_unit=unit, pos_scale=1000.0, pose_mode="T_WC"), t0, t1, T_assumed)
        out["vio"][name] = {"covered": float(ok.mean()), "N_covered": int(ok.sum())}
        if ok.any():
            r_e, t_e = geo(Rv, R_gt), np.linalg.norm(tv - t_gt, axis=1)
            rows[f"vio_{name}"] = (r_e, t_e, ok)
            out["vio"][name]["on_covered"] = summ(r_e[ok], t_e[ok])
            out["vio"][name]["inertial_on_covered"] = summ(e_gyro[ok], np.linalg.norm(t_src[main_key] - t_gt, axis=1)[ok])
            out["vio"][name]["zero_on_covered"] = summ(rows["zero_motion"][0][ok], t_zero[ok])
    for f in OUT.parent.glob("_ref_stride*.txt"):
        f.unlink()
    # the same residual IMU timestamp offsets applied to the gyroscope increment (window shifted in IMU time)
    import build_tumvi_windows as btw  # noqa: E402
    t_imu, imu = btw.read_csv(btw.room_dir(1) / "imu0" / "data.csv")
    grid, track = btw.gyro_track(t_imu, imu[:, :3])
    out["gyro_offset_sensitivity_deg"] = {}
    for d in (0.0, 0.01, 0.02, 0.05, 0.1, 0.5):
        a, b = np.clip(t0 - d, grid[0], grid[-1]), np.clip(t1 - d, grid[0], grid[-1])
        B = (track(a).inv() * track(b)).as_matrix()
        P = np.einsum("ij,njk,lk->nil", btw.R_CI, np.transpose(B, (0, 2, 1)), btw.R_CI)
        out["gyro_offset_sensitivity_deg"][f"{d:.2f}"] = float(geo(P, R_gt).mean())
    print("gyro offset sensitivity", out["gyro_offset_sensitivity_deg"])
    for kk, (r, t, m) in rows.items():
        out["rows"][kk] = summ(r[m], t[m])
        idx = np.arange(N)[::NONOVL]
        out["rows_non_overlapping"][kk] = summ(r[idx][m[idx]], t[idx][m[idx]]) if m[idx].any() else None
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("coverage", out["coverage"])
    for kk, v in out["rows"].items():
        print(f"ROW {kk:34s} rot {v['rot_mean']:.3f}/{v['rot_p50']:.3f}/{v['rot_p95']:.3f}  trans {v['trans_mean']:.2f}/{v['trans_p50']:.2f}/{v['trans_p95']:.2f}  N {v['N']}")
    print("VIO", json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ('covered', 'N_covered', 'note')} for k, v in out["vio"].items()}))
    print("DAMAGE", {k: (v["used"], round(v["max_rot_increase_deg"], 2), round(v["max_trans_increase_mm"], 1)) for k, v in out["damage"].items()})


if __name__ == "__main__":
    main()
