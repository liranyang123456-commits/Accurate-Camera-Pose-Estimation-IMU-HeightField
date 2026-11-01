#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual front ends on TUM-VI room1 (cam0 undistorted to a 512 x 512 pinhole image, f = 190 px) at every frame (20 Hz)
and at every third frame (6.7 Hz): the height-field tracker (--what hf; needs open3d), ORB-VO (essential matrix) and
ORB-HVO (homography) on ORB matches (--what orb), with the same code and configuration as on MITI and the primary rig.
Output: <imu-camera root>/tumvi_visual/<method>/room1_s<k>/traj_tum.txt (timestamps in s, camera clock).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_visual_miti as rvm  # noqa: E402

from paths import IMUCAM_ROOT  # noqa: E402
FRAMES = IMUCAM_ROOT / "tumvi_room1" / "extracted_pinhole" / "dataset-room1_512" / "mav0" / "cam0" / "data"
OUT = IMUCAM_ROOT / "tumvi_visual"
K = np.array([[190.0, 0, 256.0], [0, 190.0, 256.0], [0, 0, 1.0]])


class _Frames:
    """Adapter that serves the pinhole PNG files through the reader interface of run_visual_miti."""

    def __init__(self, stride):
        self.files = sorted(FRAMES.glob("*.png"))[::stride]

    def read(self, name):
        return name.read_bytes()


def frames(stride):
    src = _Frames(stride)
    return src, [(int(p.stem) * 1e-9, p) for p in src.files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=("hf", "orb"), required=True)
    ap.add_argument("--strides", default="1,3")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)
    rvm.OUT, rvm.K, rvm.DIST = OUT, K, np.zeros(5)
    summary = {}
    for k in (int(s) for s in args.strides.split(",")):
        rvm.frames = lambda seq, k=k: frames(k)
        seq = Path(f"room1_s{k}")
        summary[seq.name] = rvm.run_hf(seq) if args.what == "hf" else rvm.run_orb(seq)
        print(args.what, seq.name, summary[seq.name], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"summary_{args.what}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
