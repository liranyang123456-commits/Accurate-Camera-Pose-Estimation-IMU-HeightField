#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the IMU input of every 0.5 s window with the camera-IMU time offset applied.

Windows, labels, and split are unchanged: for each window [t_start, t_end] (camera time) the 60 IMU samples
are re-interpolated over [t_start + delta, t_end + delta] (IMU time), with delta estimated on the 49
training and validation sessions (eval_gyro_time_alignment.py). delta = 0 reproduces the original X.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from build_fixed_seq_dataset_from_session import (  # noqa: E402
    _interp_linear_2d, _interp_quat_nlerp, load_imu_stream_features)

ROOT = IMUCAM_ROOT
CFG = json.load(open(ROOT / "out_train_fixed_seq_20260310_031242" / "train_config.json", encoding="utf-8"))
DS = "fixed_seq_len60_win0.500s_delta_window"


def rebuild(sd: Path, delta: float):
    t, feat, quat = load_imu_stream_features(str(sd / "imu_stream.csv"))
    meta = list(csv.DictReader(open(sd / DS / "meta.csv", encoding="utf-8")))
    X = np.empty((len(meta), 60, 21), dtype=np.float32)
    for k, m in enumerate(meta):
        tq = np.linspace(float(m["t_start"]) + delta, float(m["t_end"]) + delta, num=60, dtype=np.float64)
        X[k] = np.concatenate([_interp_linear_2d(t, feat, tq), _interp_quat_nlerp(t, quat, tq)], axis=1)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=-0.50)
    ap.add_argument("--check", action="store_true", help="only verify that delta = 0 reproduces the stored X")
    args = ap.parse_args()
    sessions = []
    for item in CFG["datasets"]:
        sd = Path(item["dataset_dir"]).parent
        if sd not in sessions:
            sessions.append(sd)
    if args.check:
        worst = 0.0
        for sd in sessions[:6]:
            X0 = np.load(sd / DS / "X.npy")
            worst = max(worst, float(np.abs(rebuild(sd, 0.0) - X0).max()))
        print(f"max |X(delta=0) - X_stored| over 6 sessions: {worst:.3e}")
        return
    tag = f"{DS}_offset{args.delta:+.2f}"
    for sd in sessions:
        out = sd / tag
        out.mkdir(exist_ok=True)
        np.save(out / "X.npy", rebuild(sd, args.delta))
        for f in ("y.npy", "meta.csv"):
            shutil.copy2(sd / DS / f, out / f)
    print(f"wrote {len(sessions)} sessions to <session>/{tag}")


if __name__ == "__main__":
    main()
