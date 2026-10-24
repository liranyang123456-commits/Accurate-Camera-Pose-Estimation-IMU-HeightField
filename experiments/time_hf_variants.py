#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latency of the tracker at 640 px with default (multi-threaded) Open3D, for 8000 and 4000 points."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hf_tracker import CFG, track  # noqa: E402
from run_hf_sessions import frames_of  # noqa: E402

from paths import IMUCAM_ROOT  # noqa: E402
imgs = [img for _, img in frames_of(IMUCAM_ROOT / "Data_IMU_Camera_Pose_Sequences" / "session_20260302_013904")][:150]
out = {"threads": os.cpu_count(), "rows": []}
for max_pts in (8000, 4000):
    cfg = dict(CFG, max_pts=max_pts)
    track(imgs[:5], cfg=cfg)
    res = track(imgs, cfg=cfg, timing=True)
    row = {"width": 640, "max_pts": max_pts, "front_end_ms": float(1e3 * res["t_front"].mean()),
           "registration_ms": float(1e3 * res["t_reg"].mean()), "total_ms": float(1e3 * (res["t_front"][1:].mean() + res["t_reg"].mean()))}
    out["rows"].append(row)
    print(row, flush=True)
(HERE / "runtime_multithread.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
