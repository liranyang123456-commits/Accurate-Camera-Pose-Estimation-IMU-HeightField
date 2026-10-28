#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build 0.5 s IMU windows with camera-frame relative-pose labels from the full MITI intervention (Hartwig et al., 2022).

Source: sensordata.zip of the TUM release (IMU1 gyroscope and accelerometer at about 225-250 Hz, IR optical tracking of the
laparoscope marker L at 20 Hz) and the camera-marker extrinsics of the curated mirror (T_b_c0). Labels follow the convention
of the primary rig: R = R_{Ce<-Cs}, t = position of the start camera expressed in the end camera frame (mm), over windows
that end at every IR sample inside a continuously tracked segment. The IMU1-to-IR time offset is estimated on the training
segments by the gyroscope scan of the main paper, with a linear clock-drift term because the two clocks drift by about
0.5 ms per second over the 21 min recording. Split: the intervals covered by the nine mirror sequences (which have images)
form the test split; four further segments form the validation split; the remaining segments form the training split.
Writes miti_windows.npz (X: N x 60 x 6 [gyro deg/s, acc g], y: N x 12, t_start, t_end, segment id, split) and
miti_windows_meta.json.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT, MITI_ROOT  # noqa: E402
MITI = MITI_ROOT
SENSOR = MITI / "tum" / "sensordata.zip"
MIRROR_META = IMUCAM_ROOT / "miti_euroc" / "sequence_01" / "convert_meta.json"
ALIGN = MITI / "tum" / "mirror_time_alignment.json"
OUT = HERE / "miti_windows.npz"
OUT_META = HERE / "miti_windows_meta.json"
WIN, T, GAP_IR, GAP_IMU = 0.5, 60, 0.2, 0.05
VAL_SEGMENTS = 4  # longest non-test segments after the first two go to validation
DELTAS = np.round(np.arange(-0.60, 0.6001, 0.005), 3)


def load(z, name):
    raw = z.read(name).decode("utf-8", "ignore")
    a = np.genfromtxt(io.StringIO(raw), delimiter=",", skip_header=1)
    o = np.argsort(a[:, -1], kind="stable")
    t = a[o, -1] * 1e-3
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], a[o][keep, :-1]


def segments(t, gap):
    idx = np.split(np.arange(len(t)), np.where(np.diff(t) > gap)[0] + 1)
    return [i for i in idx if t[i[-1]] - t[i[0]] >= 2 * WIN]


def rel_pose(Re, pe, Rs, ps):
    """T_rel = inv(T_end) T_start for world<-camera poses: R_{Ce<-Cs} and the start position in the end frame."""
    return Re.T @ Rs, Re.T @ (ps - pe)


def main():
    z = zipfile.ZipFile(SENSOR)
    t_ir, q_ir = load(z, "sensordata/WIR_L_quaternion.csv")
    t_ip, p_raw = load(z, "sensordata/WIR_L_position.csv")
    # the position and quaternion messages carry stamps that differ by up to 30 ms; positions are interpolated to the quaternion stamps
    p_ir = np.stack([np.interp(t_ir, t_ip, p_raw[:, k]) for k in range(3)], 1)
    kp = np.searchsorted(t_ip, t_ir).clip(1, len(t_ip) - 1)
    good = (t_ip[kp] - t_ip[kp - 1] <= GAP_IR) & (t_ir >= t_ip[0]) & (t_ir <= t_ip[-1])
    t_ir, q_ir, p_ir = t_ir[good], q_ir[good], p_ir[good]
    t_g, gyro = load(z, "sensordata/LIMU1_angular_velocity.csv")
    t_a, acc = load(z, "sensordata/LIMU1_linear_acceleration.csv")
    R_WL = Rotation.from_quat(q_ir[:, [1, 2, 3, 0]])
    T_LC = np.asarray(json.loads(MIRROR_META.read_text(encoding="utf-8"))["T_b_c0"], dtype=np.float64)
    R_LC, p_LC = T_LC[:3, :3], T_LC[:3, 3] * 1000.0
    R_WC = (R_WL * Rotation.from_matrix(R_LC)).as_matrix()
    p_WC = p_ir + R_WL.apply(p_LC)
    segs = segments(t_ir, GAP_IR)
    align = json.loads(ALIGN.read_text(encoding="utf-8"))
    test_iv = [(v["sensordata_t0_s"], v["sensordata_t1_s"], k) for k, v in align.items()]

    # windows: end at every IR sample, start interpolated by slerp inside the same segment
    rows = []
    for si, idx in enumerate(segs):
        ts, te = t_ir[idx[0]], t_ir[idx[-1]]
        sl = Slerp(t_ir[idx], Rotation.from_matrix(R_WC[idx]))
        covered = [(a, b, k) for a, b, k in test_iv if not (b < ts or a > te)]
        for j in idx:
            t1 = t_ir[j]
            t0 = t1 - WIN
            if t0 < ts:
                continue
            k0 = int(np.searchsorted(t_ir, t0))
            if t_ir[k0] - t_ir[k0 - 1] > GAP_IR:
                continue
            R0 = sl([t0]).as_matrix()[0]
            w = (t0 - t_ir[k0 - 1]) / (t_ir[k0] - t_ir[k0 - 1])
            p0 = (1 - w) * p_WC[k0 - 1] + w * p_WC[k0]
            R, t = rel_pose(R_WC[j], p_WC[j], R0, p0)
            in_test = [k for a, b, k in covered if a <= t0 and t1 <= b]
            rows.append((si, t0, t1, R, t, in_test[0] if in_test else ""))
    seg_id = np.array([r[0] for r in rows])
    t0s = np.array([r[1] for r in rows])
    t1s = np.array([r[2] for r in rows])
    Rg = np.stack([r[3] for r in rows])
    tg_ = np.stack([r[4] for r in rows])
    test_seq = np.array([r[5] for r in rows])
    print("IR segments", len(segs), "windows", len(rows), "test windows", int((test_seq != "").sum()))

    # split by segment: test = segments overlapping a mirror sequence (windows inside the mirror interval), val = VAL_SEGMENTS longest others
    seg_len = {si: t_ir[idx[-1]] - t_ir[idx[0]] for si, idx in enumerate(segs)}
    seg_test = {si for si in seg_len if np.any((seg_id == si) & (test_seq != ""))}
    others = sorted((si for si in seg_len if si not in seg_test), key=lambda s: -seg_len[s])
    seg_val = set(others[2:2 + VAL_SEGMENTS])  # keep the two longest for training
    split = np.array(["test" if (s in seg_test and q != "") else ("drop" if s in seg_test else ("val" if s in seg_val else "train"))
                      for s, q in zip(seg_id, test_seq)])
    # windows of a test segment that lie outside the mirror interval are dropped (they are the same recording as the test data)

    # time offset: IMU1 time = IR time + delta(t), delta(t) = a + b (t - t_ref), estimated on training segments by the gyroscope scan
    H = 0.002
    tgrid = np.arange(t_g[0], t_g[-1], H)
    g_i = np.stack([np.interp(tgrid, t_g, gyro[:, k]) for k in range(3)], 1)
    dq = Rotation.from_rotvec(np.radians(g_i) * H)
    q = np.empty((len(tgrid), 4))
    x, y, zz, w = 0.0, 0.0, 0.0, 1.0
    for k, (bx, by, bz, bw) in enumerate(dq.as_quat()):
        q[k] = (x, y, zz, w)
        x, y, zz, w = (w * bx + x * bw + y * bz - zz * by, w * by - x * bz + y * bw + zz * bx, w * bz + x * by - y * bx + zz * bw,
                       w * bw - x * bx - y * by - zz * bz)
    track = Slerp(tgrid, Rotation.from_quat(q))
    R_CI = R_LC.T @ np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # IMU1 is rotated 180 deg about x relative to L (L_LIMU1 quaternion 0,1,0,0)

    def body_inc(a, b, d):
        a = np.clip(a + d, tgrid[0], tgrid[-1])
        b = np.clip(b + d, tgrid[0], tgrid[-1])
        return (track(a).inv() * track(b)).as_matrix()

    def err(d, m):
        B = body_inc(t0s[m], t1s[m], d)
        P = np.einsum("ij,njk,lk->nil", R_CI, np.transpose(B, (0, 2, 1)), R_CI)
        return np.degrees(Rotation.from_matrix(np.transpose(P, (0, 2, 1)) @ Rg[m]).magnitude())

    tr = split == "train"
    per_seg = []
    for si in sorted(set(seg_id[tr])):
        m = tr & (seg_id == si)
        if m.sum() < 40:
            continue
        sc = [(float(d), float(err(d, m).mean())) for d in DELTAS]
        d_best = min(sc, key=lambda x: x[1])
        per_seg.append({"segment": int(si), "t_mid": float(t1s[m].mean()), "n": int(m.sum()), "delta": d_best[0], "err_at_best": d_best[1],
                        "err_at_zero": dict(sc)[0.0]})
    tm = np.array([p["t_mid"] for p in per_seg])
    dd = np.array([p["delta"] for p in per_seg])
    ww = np.array([p["n"] for p in per_seg], float)
    A = np.stack([np.ones_like(tm), tm], 1)
    coef, *_ = np.linalg.lstsq(A * np.sqrt(ww)[:, None], dd * np.sqrt(ww), rcond=None)
    d_const = float(np.average(dd, weights=ww))
    print("per-segment delta:", [(p["segment"], round(p["t_mid"], 1), p["delta"]) for p in per_seg])
    print("constant delta %.4f s; linear fit a %.4f s, b %.3e s/s" % (d_const, coef[0], coef[1]))

    def delta_of(t):
        return coef[0] + coef[1] * t

    e_const = np.concatenate([err(d_const, tr & (seg_id == p["segment"])) for p in per_seg])
    e_lin = np.concatenate([err(float(delta_of(p["t_mid"])), tr & (seg_id == p["segment"])) for p in per_seg])
    e_zero = np.concatenate([err(0.0, tr & (seg_id == p["segment"])) for p in per_seg])
    print("train gyro error: delta 0 %.3f, constant %.3f, linear %.3f deg" % (e_zero.mean(), e_const.mean(), e_lin.mean()))

    # features: IMU1 gyro (deg/s) and acc (g) resampled to T samples over the shifted window
    X = np.empty((len(rows), T, 6), np.float32)
    ok = np.ones(len(rows), bool)
    for i in range(len(rows)):
        d = delta_of(t1s[i])
        tt = np.linspace(t0s[i] + d, t1s[i] + d, T)
        for src_t, src_v, c0 in ((t_g, gyro, 0), (t_a, acc, 3)):
            k0, k1 = np.searchsorted(src_t, tt[0]) - 1, np.searchsorted(src_t, tt[-1]) + 1
            if k0 < 0 or k1 >= len(src_t) or np.diff(src_t[k0:k1 + 1]).max() > GAP_IMU:
                ok[i] = False
                break
            for c in range(3):
                X[i, :, c0 + c] = np.interp(tt, src_t, src_v[:, c])
    print("windows with complete IMU1 coverage:", int(ok.sum()), "of", len(rows))
    y = np.concatenate([Rg.reshape(-1, 9), tg_], 1).astype(np.float32)
    keep = ok & (split != "drop")
    gyro_rot = np.full(len(rows), np.nan)
    R_gyro = np.repeat(np.eye(3)[None], len(rows), 0)
    for si in sorted(set(seg_id)):
        m = keep & (seg_id == si)
        if m.any():
            d_s = float(delta_of(t1s[m].mean()))
            gyro_rot[m] = err(d_s, m)
            B = body_inc(t0s[m], t1s[m], d_s)
            R_gyro[m] = np.einsum("ij,njk,lk->nil", R_CI, np.transpose(B, (0, 2, 1)), R_CI)
    np.savez_compressed(OUT, X=X[keep], y=y[keep], t_start=t0s[keep], t_end=t1s[keep], segment=seg_id[keep], split=split[keep],
                        test_sequence=test_seq[keep], gyro_rot_deg=gyro_rot[keep], R_gyro=R_gyro[keep], R_CI=R_CI, delta_coef=coef)
    meta = {"protocol": __doc__.strip().splitlines()[0], "N": int(keep.sum()),
            "split_counts": {s: int(((split == s) & keep).sum()) for s in ("train", "val", "test")},
            "segments": {s: sorted(int(x) for x in set(seg_id[(split == s) & keep])) for s in ("train", "val", "test")},
            "segment_lengths_s": {int(k): float(v) for k, v in seg_len.items()},
            "test_sequences": {k: int(((test_seq == k) & keep).sum()) for k in align},
            "delta_constant_s": d_const, "delta_linear": {"a_s": float(coef[0]), "b_s_per_s": float(coef[1])}, "per_segment_delta": per_seg,
            "train_gyro_error_deg": {"delta_zero": float(e_zero.mean()), "constant": float(e_const.mean()), "linear": float(e_lin.mean())},
            "gyro_rot_deg_by_split": {s: float(np.nanmean(gyro_rot[(split == s) & keep])) for s in ("train", "val", "test")},
            "zero_rot_deg_by_split": {s: float(np.degrees(Rotation.from_matrix(Rg[(split == s) & keep]).magnitude()).mean()) for s in ("train", "val", "test")},
            "zero_trans_mm_by_split": {s: float(np.linalg.norm(tg_[(split == s) & keep], axis=1).mean()) for s in ("train", "val", "test")},
            "windows_per_second": 20.0, "T": T, "D": 6, "channels": ["gyro_x_deg_s", "gyro_y", "gyro_z", "acc_x_g", "acc_y", "acc_z"]}
    OUT_META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("per_segment_delta", "segment_lengths_s", "segments")}, indent=1))


if __name__ == "__main__":
    main()
