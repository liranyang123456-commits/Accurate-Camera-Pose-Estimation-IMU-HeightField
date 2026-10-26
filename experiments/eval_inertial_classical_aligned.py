#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical inertial baselines of Table 5 with the corrected label convention and the camera-IMU time offset.

Rotation: gyroscope integration and onboard-quaternion increments, R_CI B^T R_CI^T, with delta and R_CI from
eval_gyro_time_alignment.py (49 training and validation sessions). Translation: zero, or accelerometer double
integration over the aligned window with the onboard attitude, zero initial velocity, and gravity estimated from
static samples of the calibration sessions (lever arm zero). Per-window errors are saved for the session statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import eval_gyro_time_alignment as ga

HERE = Path(__file__).resolve().parent
ADS = ga.DS + "_offset-0.50"
G0 = 9.80665


def stats(r, t):
    return {"N": int(r.size), "rot_mean": float(r.mean()), "rot_p50": float(np.median(r)), "rot_p95": float(np.percentile(r, 95)),
            "trans_mean": float(t.mean()), "trans_p50": float(np.median(t)), "trans_p95": float(np.percentile(t, 95)),
            "hit_rot5": float(np.mean(r <= 5.0)), "hit_trans20": float(np.mean(t <= 20.0))}


def quat_wxyz_to_R(q):
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def gravity_world(sessions):
    s, n = np.zeros(3), 0
    for sd in sessions:
        X = np.load(sd / ADS / "X.npy").astype(np.float64)
        acc, gyr, q = X[:, :, 0:3].reshape(-1, 3), X[:, :, 3:6].reshape(-1, 3), X[:, :, 17:21].reshape(-1, 4)
        m = (np.linalg.norm(gyr, axis=1) <= 3.0) & (np.abs(np.linalg.norm(acc, axis=1) - 1.0) <= 0.08)
        if m.any():
            s += np.einsum("nij,nj->i", quat_wxyz_to_R(q[m]), acc[m] * G0)
            n += int(m.sum())
    return -s / max(n, 1), n


def ins_translation(X, g_w, R_CI):
    dt = 0.5 / 59
    acc, q = X[:, :, 0:3] * G0, X[:, :, 17:21]
    Rwb = quat_wxyz_to_R(q.reshape(-1, 4)).reshape(len(X), 60, 3, 3)
    a_w = np.einsum("nkij,nkj->nki", Rwb, acc) + g_w
    v = np.zeros((len(X), 3))
    p = np.zeros((len(X), 3))
    for k in range(59):
        p += v * dt + 0.5 * a_w[:, k] * dt * dt
        v += a_w[:, k] * dt
    t_imu = -np.einsum("nji,nj->ni", Rwb[:, -1], p)
    return (t_imu @ R_CI.T) * 1000.0


def main():
    al = json.loads((HERE / "gyro_time_alignment_results.json").read_text(encoding="utf-8"))
    vg, vq = al["variants"]["gyro_corrected_convention_offset"], al["variants"]["quat_corrected_convention_offset"]
    cal, test = ga.sessions()
    te = ga.load(test)
    A = np.concatenate([s["A"] for s in te])
    y = np.concatenate([np.load(sd / ga.DS / "y.npy").astype(np.float64) for sd in test])
    t_gt = y[:, 9:12]
    names = np.concatenate([[s["name"]] * len(s["A"]) for s in te])
    r_gyro = ga.errors(A, ga.predict(te, "gyro", vg["delta_s"], True), np.asarray(vg["R_CI"]))
    r_quat = ga.errors(A, ga.predict(te, "quat", vq["delta_s"], True), np.asarray(vq["R_CI"]))
    zero_t = np.linalg.norm(t_gt, axis=1)
    g_w, n_static = gravity_world(cal)
    X = np.concatenate([np.load(sd / ADS / "X.npy").astype(np.float64) for sd in test])
    t_ins = np.linalg.norm(ins_translation(X, g_w, np.asarray(vq["R_CI"])) - t_gt, axis=1)
    out = {"protocol": __doc__.strip().splitlines()[0], "gravity_world": g_w.tolist(), "static_samples": n_static,
           "rows": {"gyro_zeroT": stats(r_gyro, zero_t), "quat_zeroT": stats(r_quat, zero_t), "quat_ins_v0": stats(r_quat, t_ins)}}
    (HERE / "inertial_classical_aligned_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez(HERE / "inertial_classical_aligned_errors.npz", session=names, rot_gyro=r_gyro, rot_quat=r_quat,
             trans_zero=zero_t, trans_ins=t_ins)
    for k, v in out["rows"].items():
        print(f"{k:12s} rot {v['rot_mean']:.3f}/{v['rot_p50']:.3f}/{v['rot_p95']:.3f} trans {v['trans_mean']:.3f}/{v['trans_p50']:.3f}/"
              f"{v['trans_p95']:.3f} hit {v['hit_rot5']:.3f}/{v['hit_trans20']:.3f} N {v['N']}")


if __name__ == "__main__":
    main()
