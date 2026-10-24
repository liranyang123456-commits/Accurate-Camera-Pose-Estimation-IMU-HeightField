#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-thread latency and memory of the height-field tracker configuration used for all visual experiments,
and of GRU6D, on 300 consecutive frames / windows of a held-out session."""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

cv2.setNumThreads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hf_tracker import CFG, track  # noqa: E402
from run_hf_sessions import frames_of  # noqa: E402

from paths import IMUCAM_ROOT  # noqa: E402
SESSION = IMUCAM_ROOT / "Data_IMU_Camera_Pose_Sequences" / "session_20260302_013904"


def main():
    imgs = [img for _, img in frames_of(SESSION)][:300]
    proc = psutil.Process()
    rows = []
    for width in (960, 640, 480):
        cfg = dict(CFG, width=width)
        track(imgs[:5], cfg=cfg)  # warm-up
        rss0 = proc.memory_info().rss
        tracemalloc.start()
        t0 = time.perf_counter()
        res = track(imgs, cfg=cfg, timing=True)
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss1 = proc.memory_info().rss
        h = int(round(imgs[0].shape[0] * width / imgs[0].shape[1]))
        rows.append({"resolution": f"{width}x{h}", "front_end_ms": float(1e3 * res["t_front"].mean()),
                     "registration_ms": float(1e3 * res["t_reg"].mean()),
                     "total_ms": float(1e3 * (res["t_front"][1:].mean() + res["t_reg"].mean())),
                     "wall_ms_per_frame": float(1e3 * wall / len(imgs)), "python_heap_peak_mb": peak / 2**20,
                     "rss_increase_mb": max(0.0, (rss1 - rss0) / 2**20), "process_rss_mb": rss1 / 2**20})
        print(rows[-1], flush=True)
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(IMUCAM_ROOT))
    from imu_camera_mlp import IMUCameraMLPTrainer
    tr = IMUCameraMLPTrainer.from_checkpoint(str(IMUCAM_ROOT / "out_train_fixed_seq_20260310_031242" / "best_model.pth"), device="cpu")
    tr.model.eval()
    X = np.load(SESSION / "fixed_seq_len60_win0.500s_delta_window" / "X.npy")[:300].astype(np.float32)
    Xs = tr.scaler_imu.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    with torch.no_grad():
        for i in range(10):
            tr.model(torch.from_numpy(Xs[i:i + 1]))
        t0 = time.perf_counter()
        for i in range(len(Xs)):
            tr.model(torch.from_numpy(Xs[i:i + 1]))
        ms = 1e3 * (time.perf_counter() - t0) / len(Xs)
    n_par = sum(p.numel() for p in tr.model.parameters())
    gru = {"ms_per_window": ms, "params": int(n_par), "checkpoint_mb": os.path.getsize(str(IMUCAM_ROOT / "out_train_fixed_seq_20260310_031242" / "best_model.pth")) / 2**20}
    print(gru)
    cpu = {"processor": __import__("platform").processor(), "cpu_count": os.cpu_count()}
    (HERE / "runtime_results.json").write_text(json.dumps({"config": CFG, "frames": len(imgs), "tracker": rows, "gru6d": gru, "machine": cpu}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
