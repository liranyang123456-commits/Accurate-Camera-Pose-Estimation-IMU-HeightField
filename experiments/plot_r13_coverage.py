#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R1-3 coverage plot: visual tracking vs IMU windows on vio_seq_20260911_010902.

This is coverage only. It is not a no-board ATE and not a fused VIO row.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEQ = Path(r"D:\reloc3r\Data_IMU_Camera_Pose_Sequences\stereo_endoscope\vio_seq_20260911_010902")
EXPORT = Path(r"D:\reloc3r\export_euroc_stereo_endoscope\vio_seq_20260911_010902")
OUT = Path(__file__).resolve().parent / "r13_coverage"
PAPER_IMG = Path(__file__).resolve().parents[1] / "images"


def _load_times(path: Path) -> np.ndarray:
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals.append(float(s) * 1e-9)
    return np.asarray(vals, np.float64)


def _load_tum(path: Path) -> np.ndarray:
    ts = []
    if not path.is_file():
        return np.zeros((0,), np.float64)
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ts.append(float(s.split()[0]))
    arr = np.asarray(ts, np.float64)
    if arr.size and np.median(arr) > 1e6:
        arr = arr * 1e-9
    return arr


def _covered(t_query: np.ndarray, t_est: np.ndarray, tol_s: float) -> np.ndarray:
    if t_est.size == 0:
        return np.zeros(t_query.shape, dtype=bool)
    idx = np.searchsorted(t_est, t_query)
    out = np.zeros(t_query.shape, dtype=bool)
    for i, t in enumerate(t_query):
        best = 1e9
        j = idx[i]
        if j < t_est.size:
            best = min(best, abs(t_est[j] - t))
        if j > 0:
            best = min(best, abs(t_est[j - 1] - t))
        out[i] = best <= tol_s
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_IMG.mkdir(parents=True, exist_ok=True)
    t_times = _load_times(EXPORT / "times.txt")
    frames = list(csv.DictReader((SEQ / "frames.csv").open(encoding="utf-8")))
    t_cam = np.array([float(r["frame_timestamp"]) for r in frames], np.float64)
    if t_cam.size != t_times.size:
        raise SystemExit(f"frame/times mismatch {t_cam.size} vs {t_times.size}")
    t0 = float(t_cam[0])
    t_rel = t_cam - t0
    t_mono = _load_tum(EXPORT / "traj_tum_orbslam3_mono.txt")
    t_mi = _load_tum(EXPORT / "traj_tum_orbslam3_mi.txt")
    vis = _covered(t_times, t_mono, 0.08)
    mi = _covered(t_times, t_mi, 0.08)

    imu_rows = list(csv.DictReader((SEQ / "imu_stream.csv").open(encoding="utf-8")))
    t_imu = np.array([float(r["timestamp"]) for r in imu_rows], np.float64)
    imu_ok = np.zeros(t_cam.shape, dtype=bool)
    for i, t in enumerate(t_cam):
        ts, te = t - 0.5, t
        if ts < t_imu[0] or te > t_imu[-1]:
            continue
        n = int(np.sum((t_imu >= ts) & (t_imu <= te)))
        imu_ok[i] = n >= 30

    summary = {
        "session": str(SEQ),
        "n_frames": int(t_cam.size),
        "duration_s": float(t_rel[-1] - t_rel[0]),
        "n_mono_traj": int(t_mono.size),
        "n_mi_traj": int(t_mi.size),
        "n_frames_mono_matched": int(vis.sum()),
        "n_frames_mi_matched": int(mi.sum()),
        "n_imu_windows": int(imu_ok.sum()),
        "mono_coverage": float(vis.mean()),
        "mi_coverage": float(mi.mean()),
        "imu_window_coverage": float(imu_ok.mean()),
        "note": "Coverage only. Not ATE. IMU windows are 0.5 s with >=30 IMU samples.",
    }
    (OUT / "coverage_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    ax.fill_between(t_rel, 0, vis.astype(float), step="post", alpha=0.55, color="#1f77b4", label=f"ORB-SLAM3 mono  {int(vis.sum())}/{t_cam.size}")
    ax.fill_between(t_rel, 1.15, 1.15 + imu_ok.astype(float) * 0.85, step="post", alpha=0.55, color="#d9480f", label=f"GRU6D IMU window  {int(imu_ok.sum())}/{t_cam.size}")
    ax.plot(t_rel, mi.astype(float) * 0.35, color="#2b8a3e", lw=1.2, label=f"ORB-SLAM3 mono-inertial  {int(mi.sum())}/{t_cam.size}")
    ax.set_ylim(-0.05, 2.15)
    ax.set_xlim(t_rel[0], t_rel[-1])
    ax.set_yticks([0.5, 1.55])
    ax.set_yticklabels(["visual track", "IMU 0.5 s window"])
    ax.set_xlabel("time (s)")
    ax.set_title("Coverage on vio_seq_20260911_010902 (not ATE)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "Fig_r13_coverage.png", dpi=180)
    fig.savefig(PAPER_IMG / "Fig_r13_coverage.png", dpi=180)
    fig.savefig(PAPER_IMG / "Fig_r13_coverage.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
