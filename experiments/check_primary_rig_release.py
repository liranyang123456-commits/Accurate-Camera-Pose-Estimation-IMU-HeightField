#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the inertial rows of Table 5 (11 held-out sessions) from the released primary-rig windows.

experiments/primary_rig/primary_rig_aligned_windows.npz holds, for the 11975 windows of the 60 sessions, the time-aligned
IMU input X (60 x 21), the checkerboard-PnP labels y (row-major R_{Ce<-Cs} and the start-camera position in the end-camera
frame, mm), the split of the evaluated GRU6D run, and the camera-frame gyroscope and onboard-quaternion rotations R_gyro and
R_quat. With X, y, and split, the GRU6D and encoder runs can be retrained with train_inertial_baselines.py.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
z = np.load(HERE / "primary_rig" / "primary_rig_aligned_windows.npz", allow_pickle=True)
split = z["split"].astype(str)
te = split == "test"
A = z["y"][te][:, :9].reshape(-1, 3, 3).astype(np.float64)
t = z["y"][te][:, 9:12].astype(np.float64)


def geo(P, Q):
    return np.degrees(Rotation.from_matrix(np.transpose(P, (0, 2, 1)) @ Q).magnitude())


def row(name, r, tr):
    print(f"{name:42s} rot {r.mean():.3f} / {np.median(r):.3f} / {np.percentile(r, 95):.3f} deg   trans {tr.mean():.3f} mm   N {r.size}")


print("windows", len(split), {s: int((split == s).sum()) for s in ("train", "val", "test")}, "sessions", len(set(z["session"])))
zero_r = np.degrees(Rotation.from_matrix(A).magnitude())
zero_t = np.linalg.norm(t, axis=1)
row("Zero motion", zero_r, zero_t)
row("Gyroscope integration (zero translation)", geo(z["R_gyro"][te].astype(np.float64), A), zero_t)
row("Onboard quaternion (zero translation)", geo(z["R_quat"][te].astype(np.float64), A), zero_t)
print("delta_s", float(z["delta_s"]), "channels", ", ".join(str(c) for c in z["channels"]))
