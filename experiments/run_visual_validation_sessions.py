#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ORB-VO or the height-field tracker on the 13 validation sessions, with the settings used for the test sessions.

The trajectories are used only to compute the variance-optimal blend weights (eval_fusion_weights_validation.py)
before the test sessions are evaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
SESS = IMUCAM_ROOT / "Data_IMU_Camera_Pose_Sequences"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=("orbvo", "hf"), required=True)
    args = ap.parse_args()
    names = json.loads((HERE / "validation_sessions.json").read_text(encoding="utf-8"))["validation"]
    if args.what == "orbvo":
        from run_classical_vo_vio_fixedtest import _write_tum, run_orb_vo_on_session
        root = IMUCAM_ROOT / "vio_runs_classical" / "orb_vo"
        for n in names:
            out = root / n / "traj_tum.txt"
            if out.exists():
                continue
            _write_tum(str(out), run_orb_vo_on_session(str(SESS / n), method="orb_vo", max_frames=0, resize_w=960))
            print("orbvo", n, flush=True)
        return
    from hf_tracker import track
    from run_hf_sessions import frames_of
    root = HERE / "hf_sessions"
    for n in names:
        d = root / n
        if (d / "traj_tum.txt").exists():
            continue
        stamps, imgs = zip(*frames_of(SESS / n))
        res = track(list(imgs))
        P = res["poses"]
        q = Rotation.from_matrix(P[:, :3, :3]).as_quat()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "traj_tum.txt", "w", encoding="utf-8") as f:
            f.write("# timestamp tx ty tz qx qy qz qw  (height-field tracker, pixel units)\n")
            for t, p, qq in zip(stamps, P[:, :3, 3], q):
                f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {qq[0]:.8f} {qq[1]:.8f} {qq[2]:.8f} {qq[3]:.8f}\n")
        print("hf", n, len(stamps), round(float(np.mean(res["fitness"])), 3), flush=True)


if __name__ == "__main__":
    main()
