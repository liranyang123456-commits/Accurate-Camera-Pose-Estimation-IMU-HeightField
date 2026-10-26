#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical inertial rotation baselines on the 11 held-out sessions with the label convention and time offset checked.

Labels are R_rel = R_end R_start^T of object->camera PnP rotations, i.e. R_{Ce<-Cs}. Gyroscope integration with
right-multiplied increments gives R_{Bs<-Be}, so the camera-frame prediction is R_CI B^T R_CI^T. The camera-IMU
time offset delta (IMU time = camera time + delta) and R_CI are estimated on the 49 training and validation
sessions only and then applied unchanged to the held-out sessions.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from build_fixed_seq_dataset_from_session import _interp_linear_2d, load_imu_stream_features  # noqa: E402

ROOT = IMUCAM_ROOT
CFG = json.load(open(ROOT / "out_train_fixed_seq_20260310_031242" / "train_config.json", encoding="utf-8"))
DS = "fixed_seq_len60_win0.500s_delta_window"
DELTAS = np.round(np.arange(-1.00, 0.2001, 0.01), 3)
H = 0.002
OUT = Path(__file__).resolve().parent / "gyro_time_alignment_results.json"


def sessions():
    test = [Path(s) for s in CFG["split"]["fixed_test"]["used_test_sessions"]]
    low = {str(s).lower() for s in test}
    seen = []
    for item in CFG["datasets"]:
        sd = Path(item["dataset_dir"]).parent
        if sd not in seen:
            seen.append(sd)
    return [s for s in seen if str(s).lower() not in low], test


def proj(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def kabsch(src, dst):
    U, _, Vt = np.linalg.svd(src.T @ dst)
    return Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T


def gyro_track(t, gyro):
    tg = np.arange(t[0], t[-1], H)
    g = np.radians(_interp_linear_2d(t, gyro, tg)) * H
    dq = Rotation.from_rotvec(g).as_quat()  # x, y, z, w
    q = np.empty((len(tg), 4))
    x, y, z, w = 0.0, 0.0, 0.0, 1.0
    for k in range(len(tg)):
        q[k] = (x, y, z, w)
        bx, by, bz, bw = dq[k]
        x, y, z, w = (w * bx + x * bw + y * bz - z * by, w * by - x * bz + y * bw + z * bx,
                      w * bz + x * by - y * bx + z * bw, w * bw - x * bx - y * by - z * bz)
    return tg, Slerp(tg, Rotation.from_quat(q))


def quat_track(t, quat_wxyz):
    _, idx = np.unique(t, return_index=True)
    tq = t[idx]
    return tq, Slerp(tq, Rotation.from_quat(quat_wxyz[idx][:, [1, 2, 3, 0]]))


def load(sess):
    out = []
    for sd in sess:
        t, feat, quat = load_imu_stream_features(str(sd / "imu_stream.csv"))
        meta = list(csv.DictReader(open(sd / DS / "meta.csv", encoding="utf-8")))
        y = np.load(sd / DS / "y.npy").astype(np.float64)
        out.append({"name": sd.name, "gyro": gyro_track(t, feat[:, 3:6]), "quat": quat_track(t, quat),
                    "ts": np.array([float(m["t_start"]) for m in meta]), "te": np.array([float(m["t_end"]) for m in meta]),
                    "A": np.stack([proj(v[:9].reshape(3, 3)) for v in y])})
    return out


def body_increment(track, ts, te, d):
    tg, sl = track
    a = np.clip(ts + d, tg[0], tg[-1])
    b = np.clip(te + d, tg[0], tg[-1])
    return (sl(a).inv() * sl(b)).as_matrix()


def predict(S, key, d, inverse):
    B = np.concatenate([body_increment(s[key], s["ts"], s["te"], d) for s in S])
    return np.transpose(B, (0, 2, 1)) if inverse else B


def errors(A, B, X):
    M = np.einsum("ij,njk,lk->nil", X, B, X)
    return np.degrees(Rotation.from_matrix(np.transpose(M, (0, 2, 1)) @ A).magnitude())


def fit(S, key, d, inverse):
    A = np.concatenate([s["A"] for s in S])
    B = predict(S, key, d, inverse)
    X = kabsch(Rotation.from_matrix(B).as_rotvec(), Rotation.from_matrix(A).as_rotvec())
    return X, float(errors(A, B, X).mean())


def summary(e):
    return {"mean": float(e.mean()), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "hit_rot5": float(np.mean(e <= 5.0)), "N": int(e.size)}


def main():
    tr_s, te_s = sessions()
    tr, te = load(tr_s), load(te_s)
    A_te = np.concatenate([s["A"] for s in te])
    names = np.concatenate([[s["name"]] * len(s["A"]) for s in te])
    out = {"protocol": __doc__.strip().splitlines()[0], "n_calibration_sessions": len(tr), "n_test_sessions": len(te),
           "zero_rotation": summary(np.degrees(Rotation.from_matrix(A_te).magnitude())), "variants": {}, "scan": {}}
    for key in ("gyro", "quat"):
        X0, _ = fit(tr, key, 0.0, False)
        out["variants"][f"{key}_paper_convention_no_offset"] = summary(errors(A_te, predict(te, key, 0.0, False), X0))
        X1, _ = fit(tr, key, 0.0, True)
        out["variants"][f"{key}_corrected_convention_no_offset"] = summary(errors(A_te, predict(te, key, 0.0, True), X1))
        scan = []
        for d in DELTAS:
            X, e_tr = fit(tr, key, float(d), True)
            scan.append((float(d), e_tr, X))
        d_best, e_best, X_best = min(scan, key=lambda r: r[1])
        e = errors(A_te, predict(te, key, d_best, True), X_best)
        v = summary(e)
        v.update({"delta_s": d_best, "calibration_mean": e_best, "R_CI": X_best.tolist(),
                  "per_session_mean": {n: float(e[names == n].mean()) for n in dict.fromkeys(names.tolist())}})
        out["variants"][f"{key}_corrected_convention_offset"] = v
        out["scan"][key] = {"delta_s": [r[0] for r in scan], "calibration_mean": [r[1] for r in scan]}
        if key == "gyro":
            per = []
            for s in tr + te:
                sc = [errors(s["A"], np.transpose(body_increment(s["gyro"], s["ts"], s["te"], float(d)), (0, 2, 1)), X_best).mean()
                      for d in DELTAS]
                per.append({"session": s["name"], "split": "test" if s in te else "calibration", "n": int(len(s["A"])),
                            "best_delta_s": float(DELTAS[int(np.argmin(sc))])})
            bd = np.array([p["best_delta_s"] for p in per])
            out["per_session_offset"] = {"median_s": float(np.median(bd)), "iqr_s": [float(x) for x in np.percentile(bd, [25, 75])],
                                         "sessions": per}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("zero rotation", out["zero_rotation"])
    for k, v in out["variants"].items():
        print(f"{k:40s} mean {v['mean']:.3f} median {v['median']:.3f} p95 {v['p95']:.3f} hit5 {v['hit_rot5']:.3f}"
              + (f" delta {v['delta_s']:+.2f}s" if "delta_s" in v else ""))
    print("per-session offset median/IQR:", out["per_session_offset"]["median_s"], out["per_session_offset"]["iqr_s"])


if __name__ == "__main__":
    main()
