#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build 0.5 s IMU windows with camera-frame relative-pose labels for the TUM-VI rooms 1-6 (512 px, calibrated release).

The reference is the motion-capture pose of the IMU (time-aligned with the IMU in the calibrated release); camera poses
follow from the Kalibr extrinsics T_b_c1 of the ORB-SLAM3 TUM-VI configuration. Labels follow the convention of the
primary rig and of MITI: R = R_{Ce<-Cs}, t = position of the start camera in the end camera frame (mm). Windows end at
every camera frame of room1 (20 Hz) and on a 20 Hz grid in the other rooms. Split: room1 test, room6 validation, rooms
2-5 training. The camera--IMU offset and R_CI are also estimated on the training rooms with the procedure of the primary
rig (gyroscope scan and Procrustes), with the original timestamps and with a 0.50 s offset injected into the IMU
timestamps, and compared with the hardware synchronization and the Kalibr rotation. Writes tumvi_windows.npz and
tumvi_windows_meta.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT, TUMVI_ROOT  # noqa: E402
ROOM1 = IMUCAM_ROOT / "tumvi_room1" / "extracted" / "dataset-room1_512" / "mav0"
OTHER = TUMVI_ROOT / "extracted"
OUT = HERE / "tumvi_windows.npz"
OUT_META = HERE / "tumvi_windows_meta.json"
WIN, T, GAP_GT, GAP_IMU, RATE = 0.5, 60, 0.05, 0.05, 20.0
SPLIT = {1: "test", 2: "train", 3: "train", 4: "train", 5: "train", 6: "val"}
T_B_C = np.array([[-0.9995250378696743, 0.0075019185074052044, -0.02989013031643309, 0.045574835649698026],
                  [0.029615343885863205, -0.03439736061393144, -0.998969345370175, -0.071161801837997044],
                  [-0.008522328211654736, -0.9993800792498829, 0.03415885127385616, -0.044681254117144367],
                  [0.0, 0.0, 0.0, 1.0]])
R_IC, P_IC = T_B_C[:3, :3], T_B_C[:3, 3] * 1000.0
R_CI = R_IC.T
DELTAS = np.round(np.arange(-1.0, 1.0001, 0.01), 2)


def room_dir(r):
    return ROOM1 if r == 1 else OTHER / f"dataset-room{r}_512" / "mav0"


def read_csv(path):
    a = np.genfromtxt(path, delimiter=",", skip_header=1)
    o = np.argsort(a[:, 0], kind="stable")
    a = a[o]
    keep = np.r_[True, np.diff(a[:, 0]) > 0]
    return a[keep, 0] * 1e-9, a[keep, 1:]


def rel_pose(Re, pe, Rs, ps):
    return Re.T @ Rs, Re.T @ (ps - pe)


def gyro_track(t_imu, w):
    """Right-multiplied integration of the gyroscope on a 2 ms grid: orientation of the IMU relative to its first sample."""
    H = 0.002
    grid = np.arange(t_imu[0], t_imu[-1], H)
    wi = np.stack([np.interp(grid, t_imu, w[:, k]) for k in range(3)], 1)
    q = Rotation.identity()
    out = [q]
    for k in range(len(grid) - 1):
        q = q * Rotation.from_rotvec(wi[k] * H)
        out.append(q)
    return grid, Slerp(grid, Rotation.concatenate(out))


def windows_for_room(r):
    d = room_dir(r)
    t_imu, imu = read_csv(d / "imu0" / "data.csv")
    t_gt, gt = read_csv(d / "state_groundtruth_estimate0" / "data.csv")
    R_WI = Rotation.from_quat(gt[:, [4, 5, 6, 3]])
    p_WI = gt[:, :3] * 1000.0
    R_WC = (R_WI * Rotation.from_matrix(R_IC)).as_matrix()
    p_WC = p_WI + R_WI.apply(P_IC)
    sl = Slerp(t_gt, Rotation.from_matrix(R_WC))
    if r == 1:
        ends = np.array(sorted(int(p.stem) for p in (d / "cam0" / "data").glob("*.png"))) * 1e-9
    else:
        ends = np.arange(t_gt[0] + WIN, t_gt[-1], 1.0 / RATE)
    rows = []
    for t1 in ends:
        t0 = t1 - WIN
        if t0 < t_gt[0] or t1 > t_gt[-1] or t0 < t_imu[0] + 1.1 or t1 > t_imu[-1] - 1.1:
            continue
        k0, k1 = np.searchsorted(t_gt, [t0, t1])
        if k0 == 0 or k1 == 0 or k1 >= len(t_gt) or t_gt[k0] - t_gt[k0 - 1] > GAP_GT or t_gt[k1] - t_gt[k1 - 1] > GAP_GT:
            continue
        Rs, Re = sl([t0, t1]).as_matrix()
        ps = np.array([np.interp(t0, t_gt, p_WC[:, k]) for k in range(3)])
        pe = np.array([np.interp(t1, t_gt, p_WC[:, k]) for k in range(3)])
        R, t = rel_pose(Re, pe, Rs, ps)
        rows.append((t0, t1, R, t))
    return t_imu, imu, rows


def main():
    data = {}
    for r in SPLIT:
        t_imu, imu, rows = windows_for_room(r)
        data[r] = (t_imu, imu, rows)
        print(f"room{r}: {len(rows)} windows, IMU {len(t_imu)} samples, {t_imu[-1] - t_imu[0]:.1f} s", flush=True)
    tracks = {r: gyro_track(data[r][0], data[r][1][:, :3]) for r in data}

    def body_inc(r, t0, t1, d):
        grid, tr = tracks[r]
        a = np.clip(t0 + d, grid[0], grid[-1])
        b = np.clip(t1 + d, grid[0], grid[-1])
        return (tr(a).inv() * tr(b)).as_matrix()

    def labels(r):
        rows = data[r][2]
        return np.array([x[0] for x in rows]), np.array([x[1] for x in rows]), np.stack([x[2] for x in rows]), np.stack([x[3] for x in rows])

    def gyro_err(r, d, R_ci, inject=0.0):
        t0, t1, Rg, _ = labels(r)
        B = body_inc(r, t0 - inject, t1 - inject, d)  # injected offset: IMU stamps are later by `inject`
        P = np.einsum("ij,njk,lk->nil", R_ci, np.transpose(B, (0, 2, 1)), R_ci)
        return np.degrees(Rotation.from_matrix(np.transpose(P, (0, 2, 1)) @ Rg).magnitude())

    def procrustes(rooms, d, inject=0.0):
        A, Bv = [], []
        for r in rooms:
            t0, t1, Rg, _ = labels(r)
            Bm = body_inc(r, t0 - inject, t1 - inject, d)
            A.append(Rotation.from_matrix(np.transpose(Bm, (0, 2, 1))).as_rotvec())
            Bv.append(Rotation.from_matrix(Rg).as_rotvec())
        A, Bv = np.concatenate(A), np.concatenate(Bv)
        U, _, Vt = np.linalg.svd(A.T @ Bv)
        Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        return Vt.T @ Dm @ U.T

    train = [r for r, s in SPLIT.items() if s == "train"]
    calib = {}
    for name, inject in (("original_timestamps", 0.0), ("imu_offset_injected_0.50s", 0.50)):
        scan = []
        for d in DELTAS:
            Rci = procrustes(train, d, inject)
            e = np.concatenate([gyro_err(r, d, Rci, inject) for r in train])
            scan.append((float(d), float(e.mean()), Rci))
        best = min(scan, key=lambda x: x[1])
        ang = float(np.degrees(Rotation.from_matrix(best[2].T @ R_CI).magnitude()))
        calib[name] = {"delta_s": best[0], "gyro_err_deg_at_best": best[1], "angle_to_kalibr_R_CI_deg": ang,
                       "gyro_err_deg_at_zero_kalibr": float(np.concatenate([gyro_err(r, 0.0, R_CI, inject) for r in train]).mean())}
        print(name, calib[name], flush=True)

    X_all, y_all, t0_all, t1_all, room_all, split_all, gyro_all, Rg_all = [], [], [], [], [], [], [], []
    for r in SPLIT:
        t_imu, imu, rows = data[r]
        t0, t1, Rg, tg = labels(r)
        ok = np.ones(len(rows), bool)
        X = np.empty((len(rows), T, 6), np.float32)
        gyro_deg, acc_g = np.degrees(imu[:, :3]), imu[:, 3:] / 9.80665
        for i in range(len(rows)):
            tt = np.linspace(t0[i], t1[i], T)
            k0, k1 = np.searchsorted(t_imu, tt[0]) - 1, np.searchsorted(t_imu, tt[-1]) + 1
            if k0 < 0 or k1 >= len(t_imu) or np.diff(t_imu[k0:k1 + 1]).max() > GAP_IMU:
                ok[i] = False
                continue
            for c in range(3):
                X[i, :, c] = np.interp(tt, t_imu, gyro_deg[:, c])
                X[i, :, 3 + c] = np.interp(tt, t_imu, acc_g[:, c])
        B = body_inc(r, t0, t1, 0.0)
        P = np.einsum("ij,njk,lk->nil", R_CI, np.transpose(B, (0, 2, 1)), R_CI)
        e = np.degrees(Rotation.from_matrix(np.transpose(P, (0, 2, 1)) @ Rg).magnitude())
        X_all.append(X[ok]); y_all.append(np.concatenate([Rg.reshape(-1, 9), tg], 1)[ok].astype(np.float32))
        t0_all.append(t0[ok]); t1_all.append(t1[ok]); room_all.append(np.full(ok.sum(), r)); split_all.append(np.full(ok.sum(), SPLIT[r]))
        gyro_all.append(e[ok]); Rg_all.append(P[ok])
    X, y = np.concatenate(X_all), np.concatenate(y_all)
    room, split = np.concatenate(room_all), np.concatenate(split_all)
    test_seq = np.where(split == "test", "room1", "")
    gyro_rot = np.concatenate(gyro_all)
    np.savez_compressed(OUT, X=X, y=y, t_start=np.concatenate(t0_all), t_end=np.concatenate(t1_all), room=room, split=split,
                        test_sequence=test_seq, gyro_rot_deg=gyro_rot, R_gyro=np.concatenate(Rg_all), R_CI=R_CI, T_b_c=T_B_C)
    zr = np.degrees(Rotation.from_matrix(y[:, :9].reshape(-1, 3, 3)).magnitude())
    zt = np.linalg.norm(y[:, 9:12], axis=1)
    meta = {"protocol": __doc__.strip().splitlines()[0], "N": int(len(y)),
            "split_counts": {s: int((split == s).sum()) for s in ("train", "val", "test")},
            "rooms": {f"room{r}": {"split": SPLIT[r], "N": int((room == r).sum())} for r in SPLIT},
            "calibration_check": calib,
            "gyro_rot_deg_by_split": {s: float(gyro_rot[split == s].mean()) for s in ("train", "val", "test")},
            "zero_rot_deg_by_split": {s: float(zr[split == s].mean()) for s in ("train", "val", "test")},
            "zero_trans_mm_by_split": {s: float(zt[split == s].mean()) for s in ("train", "val", "test")},
            "windows_per_second": RATE, "T": T, "D": 6, "channels": ["gyro_x_deg_s", "gyro_y", "gyro_z", "acc_x_g", "acc_y", "acc_z"]}
    OUT_META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
