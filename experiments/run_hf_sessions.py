#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the height-field tracker on every frame of the 11 held-out sessions; write TUM trajectories."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from hf_tracker import CFG, track  # noqa: E402
from eval_vio_fixedtest_aggregate import _load_fixed_test_sessions_from_train_config  # noqa: E402

RUN = IMUCAM_ROOT / "out_train_fixed_seq_20260310_031242"
OUT = Path(__file__).resolve().parent / "hf_sessions"


def frames_of(session: Path):
    ts = {}
    with open(session / "frames.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts[int(float(r["frame_idx"]))] = float(r["frame_timestamp"])
    cap = cv2.VideoCapture(str(session / "video.avi"))
    i = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if i in ts:
            yield ts[i], img
    cap.release()


def main():
    OUT.mkdir(exist_ok=True)
    info = {"config": CFG}
    for sd in _load_fixed_test_sessions_from_train_config(str(RUN / "train_config.json")):
        sd = Path(sd)
        stamps, imgs = [], []
        for t, img in frames_of(sd):
            stamps.append(t)
            imgs.append(img)
        res = track(imgs)
        P = res["poses"]
        q = Rotation.from_matrix(P[:, :3, :3]).as_quat()  # x y z w
        d = OUT / sd.name
        d.mkdir(exist_ok=True)
        with open(d / "traj_tum.txt", "w", encoding="utf-8") as f:
            f.write("# timestamp tx ty tz qx qy qz qw  (height-field tracker, pixel units)\n")
            for t, p, qq in zip(stamps, P[:, :3, 3], q):
                f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {qq[0]:.8f} {qq[1]:.8f} {qq[2]:.8f} {qq[3]:.8f}\n")
        fit = res["fitness"]
        info[sd.name] = {"frames": len(stamps), "fitness_mean": float(fit.mean()), "fitness_lt_0.2": int((fit < 0.2).sum())}
        print(sd.name, info[sd.name], flush=True)
    (OUT / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
