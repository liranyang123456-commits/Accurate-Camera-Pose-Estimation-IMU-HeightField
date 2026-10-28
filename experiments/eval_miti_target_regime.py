#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-regime check with recorded IMU data: MITI laparoscopic sequence 01 (Hartwig et al., 2022).

The sequence provides a handheld stereo laparoscope (cam0 used, 960 x 540, about 59 Hz), a 200 Hz IMU, the camera-IMU
extrinsics, and reference poses from an infrared optical tracker that is independent of the IMU. Over 0.5 s windows
ending at every frame, the script evaluates the zero-rotation estimate, gyroscope integration mapped to the camera frame
with the calibrated extrinsics, the height-field increments, and the consistency-gated geodesic blend of the two
rotations (tau_rot = 3 deg) with gamma_R = 0.05 (validation weight of the checkerboard sessions), 0.5 (default), and
without the gate. No inertial translation is available for this mounting (GRU6D is trained for the primary rig), so
translation is reported for the height field with the reference scale and for the zero-translation estimate only.

  --run-hf   run the height-field tracker (hf_tracker.py; needs open3d) on cam0 and write the TUM trajectory first.
  --stride k use every k-th frame (k = 10 gives 5.9 Hz, the frame rate of the primary rig): the tracker chains k times fewer
             registrations per 0.5 s window, and the windows end at the subsampled frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))

MITI = IMUCAM_ROOT / "miti_euroc" / "sequence_01"
HF_DIR = IMUCAM_ROOT / "miti_hf_runs" / "sequence_01"
OUT_BASE = HERE / "miti_target_regime_results"
WIN, TAU = 0.5, 3.0
DELTAS = np.round(np.arange(-0.30, 0.3001, 0.01), 3)


def frame_times():
    rows = [ln.split(",") for ln in (MITI / "mav0" / "cam0" / "data.csv").read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    return np.array([int(r[0]) for r in rows], dtype=np.int64)


def load_gt():
    g = np.loadtxt(MITI / "groundtruth_euroc.csv", delimiter=",", skiprows=1)
    t = g[:, 0] * 1e-9
    p = g[:, 1:4]
    R = Rotation.from_quat(g[:, [5, 6, 7, 4]]).as_matrix()  # file order w, x, y, z
    return t, p, R


def run_hf(stride=1):
    import cv2
    from hf_tracker import CFG, static_mask, track  # noqa: E402  (needs open3d)
    ts = frame_times()[::stride]
    HF_TRAJ = HF_DIR / ("traj_tum.txt" if stride == 1 else f"traj_tum_stride{stride}.txt")
    files = [MITI / "mav0" / "cam0" / "data" / f"{t}.png" for t in ts]
    mask = static_mask([cv2.imread(str(f)) for f in files[::40]])
    out = track((cv2.imread(str(f)) for f in files), mask, CFG)
    P = out["poses"]
    q = Rotation.from_matrix(P[:, :3, :3]).as_quat()
    HF_TRAJ.parent.mkdir(parents=True, exist_ok=True)
    with open(HF_TRAJ, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw  (height-field tracker, pixel units)\n")
        for t, p, qq in zip(ts * 1e-9, P[:, :3, 3], q):
            f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {qq[0]:.8f} {qq[1]:.8f} {qq[2]:.8f} {qq[3]:.8f}\n")
    print("wrote", HF_TRAJ, "frames", len(P), "mean fitness", float(out["fitness"].mean()))


def summ(e):
    e = np.asarray(e, float)
    return {"mean": float(e.mean()), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)), "N": int(e.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-hf", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    stride = max(1, args.stride)
    HF_TRAJ = HF_DIR / ("traj_tum.txt" if stride == 1 else f"traj_tum_stride{stride}.txt")
    OUT = Path(str(OUT_BASE) + ("" if stride == 1 else f"_stride{stride}") + ".json")
    if args.run_hf:
        run_hf(stride)
        return
    from eval_vio_trajectory_rpe_ate import PoseSeq, _interp_pose, _rel_end_frame  # noqa: E402
    import eval_fixedtest_aligned as ev  # noqa: E402
    import eval_gyro_time_alignment as ga  # noqa: E402

    meta = json.loads((MITI / "convert_meta.json").read_text(encoding="utf-8"))
    T_BC = np.asarray(meta["T_b_c0"], dtype=np.float64)
    tg, pg, Rg = load_gt()
    # camera poses in the world frame, positions in mm
    Rc = Rg @ T_BC[:3, :3]
    pc = (pg + np.einsum("nij,j->ni", Rg, T_BC[:3, 3])) * 1000.0
    cam_seq = PoseSeq(t=tg, p=pc, q=Rotation.from_matrix(Rc).as_quat()[:, [3, 0, 1, 2]])
    body_seq = PoseSeq(t=tg, p=pg * 1000.0, q=Rotation.from_matrix(Rg).as_quat()[:, [3, 0, 1, 2]])

    ts_cam = frame_times()[::stride] * 1e-9
    recs = []
    for te in ts_cam:
        ts = te - WIN
        e, s = _interp_pose(cam_seq, te, max_gap_s=0.1), _interp_pose(cam_seq, ts, max_gap_s=0.1)
        eb, sb = _interp_pose(body_seq, te, max_gap_s=0.1), _interp_pose(body_seq, ts, max_gap_s=0.1)
        if e is None or s is None or eb is None or sb is None:
            continue
        R, t = _rel_end_frame(e[0], e[1], s[0], s[1])
        Rb, tb = _rel_end_frame(eb[0], eb[1], sb[0], sb[1])
        recs.append({"ts": ts, "te": te, "R_gt": R, "t_gt": t.reshape(3), "R_gt_body": Rb})
    N = len(recs)
    ts_arr = np.array([r["ts"] for r in recs])
    te_arr = np.array([r["te"] for r in recs])
    R_gt = np.stack([r["R_gt"] for r in recs])
    t_gt = np.stack([r["t_gt"] for r in recs])
    dt = float(np.median(np.diff(ts_cam)))
    stride = max(1, int(round(WIN / dt)))

    imu = np.loadtxt(MITI / "mav0" / "imu0" / "data.csv", delimiter=",", skiprows=1)
    track_ = ga.gyro_track(imu[:, 0] * 1e-9, np.degrees(imu[:, 1:4]))
    R_CI = T_BC[:3, :3].T  # IMU/body-to-camera rotation from the calibration

    def gyro_pred(delta, X=R_CI):
        B = ga.body_increment(track_, ts_arr, te_arr, delta)
        return np.einsum("ij,njk,lk->nil", X, np.transpose(B, (0, 2, 1)), X)

    def err(P, A=R_gt):
        return np.degrees(Rotation.from_matrix(np.transpose(P, (0, 2, 1)) @ A).magnitude())

    # frame hypotheses and time offset (diagnostics; the evaluation below uses the calibrated R_CI and delta = 0)
    e0 = err(gyro_pred(0.0))
    B0 = ga.body_increment(track_, ts_arr, te_arr, 0.0)
    e_body = err(np.transpose(B0, (0, 2, 1)), np.stack([r["R_gt_body"] for r in recs]))
    X_fit = ga.kabsch(Rotation.from_matrix(np.transpose(B0, (0, 2, 1))).as_rotvec(), Rotation.from_matrix(R_gt).as_rotvec())
    scan = [(float(d), float(err(gyro_pred(float(d))).mean())) for d in DELTAS]
    d_star = min(scan, key=lambda x: x[1])
    diag = {"N_windows": N, "window_stride_frames": stride, "N_non_overlapping": int(len(range(0, N, stride))),
            "gyro_calibrated_R_CI_delta0_mean": float(e0.mean()),
            "gyro_vs_body_reference_identity_mean": float(e_body.mean()),
            "procrustes_R_CI_angle_to_calibrated_deg": float(np.degrees(Rotation.from_matrix(X_fit @ R_CI.T).magnitude())),
            "gyro_procrustes_R_CI_delta0_mean": float(err(gyro_pred(0.0, X_fit)).mean()),
            "delta_scan_best_s": d_star[0], "delta_scan_best_mean": d_star[1], "delta_scan_zero_mean": dict(scan)[0.0],
            "gt_rotation_per_window_deg": summ(err(np.repeat(np.eye(3)[None], N, 0))),
            "gt_translation_per_window_mm": summ(np.linalg.norm(t_gt, axis=1))}

    out = {"protocol": __doc__.strip().splitlines()[0], "sequence": MITI.name, "duration_s": float(ts_cam[-1] - ts_cam[0]),
           "frame_stride": stride, "frame_rate_hz": 1.0 / dt, "diagnostics": diag, "rows": {}, "rows_non_overlapping": {}}
    R_gyro = gyro_pred(0.0)
    rows = {"zero_rotation": err(np.repeat(np.eye(3)[None], N, 0)), "gyro": e0}

    est, scale = ev.load_vis(HF_TRAJ, cam_seq)
    vis_ok = np.zeros(N, bool)
    R_vis = np.repeat(np.eye(3)[None], N, 0)
    t_vis = np.zeros((N, 3))
    if est is not None:
        for i, r in enumerate(recs):
            e, s = _interp_pose(est, r["te"], max_gap_s=0.2), _interp_pose(est, r["ts"], max_gap_s=0.2)
            if e is not None and s is not None:
                R, tv = _rel_end_frame(e[0], e[1], s[0], s[1])
                vis_ok[i], R_vis[i], t_vis[i] = True, R, tv.reshape(3)
    out["hf_reference_scale"] = float(scale)
    out["hf_support_fraction"] = float(vis_ok.mean())
    cons = np.array([vis_ok[i] and ev.geo(R_vis[i], R_gyro[i]) <= TAU for i in range(N)])
    out["gate_accept_fraction"] = float(cons.mean())

    def blend_rows(gR, gate):
        R = np.array([ev.slerp(R_gyro[i], R_vis[i], gR) if (vis_ok[i] and (not gate or cons[i])) else R_gyro[i] for i in range(N)])
        e = err(R)
        return e, float((e - e0).max())

    for name, (gR, gate) in {"blend_gR0.05_gated": (0.05, True), "blend_gR0.5_gated": (0.5, True), "blend_gR1.0_gated": (1.0, True),
                             "blend_gR0.5_ungated": (0.5, False)}.items():
        e, mx = blend_rows(gR, gate)
        rows[name] = e
        out["rows"][name + "_max_excess_deg"] = mx
    if vis_ok.any():
        rows["hf_only_supported"] = err(R_vis[vis_ok], R_gt[vis_ok])
        rows["gyro_on_supported"] = e0[vis_ok]
        rows["zero_on_supported"] = rows["zero_rotation"][vis_ok]
    for k, e in rows.items():
        out["rows"][k] = summ(e)
        out["rows_non_overlapping"][k] = summ(e[::stride]) if e.size == N else summ(e[::max(1, int(round(stride * e.size / N)))])
    out["translation_mm"] = {"zero_translation": summ(np.linalg.norm(t_gt, axis=1))}
    if vis_ok.any():
        out["translation_mm"]["hf_reference_scale_supported"] = summ(np.linalg.norm(t_vis[vis_ok] * scale - t_gt[vis_ok], axis=1))
        out["translation_mm"]["zero_translation_supported"] = summ(np.linalg.norm(t_gt[vis_ok], axis=1))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("DIAG", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in diag.items() if not isinstance(v, dict)})
    print("GT motion per window:", diag["gt_rotation_per_window_deg"], diag["gt_translation_per_window_mm"])
    print("hf scale", round(scale, 4), "support", round(out["hf_support_fraction"], 3), "gate accept", round(out["gate_accept_fraction"], 3))
    for k, v in out["rows"].items():
        if isinstance(v, dict):
            print(f"ROW {k:24s} mean {v['mean']:.3f} median {v['median']:.3f} p95 {v['p95']:.3f} N {v['N']}")
        else:
            print(f"ROW {k:24s} {v:.3f}")
    print("NON-OVERLAPPING", {k: round(v["mean"], 3) for k, v in out["rows_non_overlapping"].items()})
    print("TRANS", {k: round(v["mean"], 3) for k, v in out["translation_mm"].items()})


if __name__ == "__main__":
    main()
