#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weak-B qualitative eval: chain old GRU6D relative poses along a session.

If chessboard PnP exists, windows are colored board-visible vs not.
This does not invent metric ATE for the no-board segment.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(r"D:\reloc3r")
CKPT = ROOT / r"out_train_fixed_seq_20260310_031242\best_model.pth"
LABEL_PY = ROOT / "label_session_chessboard_pose_offline.py"
CALIB = (
    ROOT
    / r"Data_IMU_Camera_Pose_Sequences\stereo_endoscope\calib_intrinsics_20260911_005642\camera_calibration_cam0.json"
)
OUT_ROOT = Path(__file__).resolve().parent / "gru6d_weak_b"


def _rel_chain(R_list: np.ndarray, t_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chain window-level camera-relative poses into a trajectory."""
    T = np.eye(4)
    pos = [T[:3, 3].copy()]
    for R, t in zip(R_list, t_list):
        d = np.eye(4)
        d[:3, :3] = R
        d[:3, 3] = t
        T = T @ d
        pos.append(T[:3, 3].copy())
    return np.stack(pos, 0), T


def _project_so3(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    Rp = u @ vt
    if np.linalg.det(Rp) < 0:
        u[:, -1] *= -1
        Rp = u @ vt
    return Rp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", required=True)
    ap.add_argument("--skip_pnp", action="store_true")
    args = ap.parse_args()
    session_dir = Path(args.session_dir)
    sys.path.insert(0, str(ROOT))
    from build_fixed_seq_dataset_from_session import (
        build_dataset,
        load_frames_with_pose,
        load_imu_stream_features,
    )
    from imu_camera_mlp import IMUCameraMLPTrainer

    video = session_dir / "cam0" / "video.avi"
    if not video.is_file():
        video = session_dir / "video.avi"
    frames_labeled = session_dir / "frames_labeled.csv"
    if not args.skip_pnp and video.is_file() and not frames_labeled.is_file():
        cmd = [
            sys.executable,
            str(LABEL_PY),
            "--session_dir",
            str(session_dir),
            "--video_path",
            str(video),
            "--camera_calib_json",
            str(CALIB),
            "--out_frames_csv",
            str(frames_labeled),
            "--downscale",
            "0.5",
            "--max_reproj_rmse",
            "2.0",
        ]
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)

    imu_csv = session_dir / "imu_stream.csv"
    t_imu, feat17, quat4 = load_imu_stream_features(str(imu_csv))
    n_pnp_any = 0
    n_pnp_12x9 = 0
    t_valid = np.zeros((0,), np.float64)
    n_frames = 0
    if frames_labeled.is_file():
        with frames_labeled.open(newline="", encoding="utf-8") as f:
            labeled_rows = list(csv.DictReader(f))
        n_frames = len(labeled_rows)
        n_pnp_any = sum(
            1 for r in labeled_rows if str(r.get("pose_valid", "0")).strip() in ("1", "true", "True")
        )
        t_valid_list = []
        for r in labeled_rows:
            if str(r.get("pose_valid", "0")).strip() not in ("1", "true", "True"):
                continue
            name = str(r.get("chessboard_name", "")).strip()
            if name not in ("", "12x9-3mm"):
                continue
            t_valid_list.append(float(r["frame_timestamp"]))
        n_pnp_12x9 = len(t_valid_list)
        t_valid = np.asarray(t_valid_list, np.float64)
        try:
            load_frames_with_pose(
                str(frames_labeled), pose_valid_only=True, max_reproj_rmse=2.0
            )
        except Exception as e:
            print("pose load failed:", e)

    # Always also build a dense IMU-only window grid at 10 Hz over the IMU span
    t0 = float(t_imu[0] + 0.5)
    t1 = float(t_imu[-1])
    t_ends = np.arange(t0, t1, 0.1)
    # Fake FramePose-like ends: we only need IMU windows for qualitative chain.
    # If labeled poses exist, reuse builder for GT-aligned windows; else interpolate IMU only.
    trainer = IMUCameraMLPTrainer.from_checkpoint(str(CKPT))
    scaler = trainer.scaler_imu
    sp = trainer.scaler_pose

    def interp_windows(t_ends_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        used = []
        for te in t_ends_arr:
            ts = te - 0.5
            tq = np.linspace(ts, te, 60)
            if ts < t_imu[0] or te > t_imu[-1]:
                continue
            feat = np.stack([np.interp(tq, t_imu, feat17[:, j]) for j in range(feat17.shape[1])], 1)
            q = np.stack([np.interp(tq, t_imu, quat4[:, j]) for j in range(4)], 1)
            n = np.linalg.norm(q, axis=1, keepdims=True)
            q = q / np.clip(n, 1e-8, None)
            x = np.concatenate([feat, q], 1).astype(np.float32)
            xs.append(x)
            used.append(float(te))
        if not xs:
            return np.zeros((0, 60, 21), np.float32), np.zeros((0,), np.float64)
        return np.stack(xs, 0), np.asarray(used, np.float64)

    X, t_ends_used = interp_windows(t_ends)
    if X.shape[0] == 0:
        raise SystemExit("no IMU windows")
    Xs = scaler.transform(X.reshape(-1, 21)).reshape(X.shape).astype(np.float32)
    device = trainer.device
    trainer.model.eval()
    preds = []
    with torch.no_grad():
        xb = torch.from_numpy(Xs).to(device)
        for i in range(0, xb.shape[0], 64):
            pred = trainer.model(xb[i : i + 64]).cpu().numpy()
            preds.append(pred)
    pred = np.concatenate(preds, 0)
    R = pred[:, :9].reshape(-1, 3, 3)
    t = pred[:, 9:12].copy()
    t = t * np.asarray(sp.t_scale_, dtype=np.float32) + np.asarray(sp.t_mean_, dtype=np.float32)
    R = np.stack([_project_so3(r) for r in R], 0)
    rot_deg = np.degrees(np.arccos(np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) * 0.5, -1, 1)))
    trans_n = np.linalg.norm(t, axis=1)
    pos, _ = _rel_chain(R, t)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tag = session_dir.name
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if t_valid.size:
        t_last_valid = float(t_valid.max())
        board_mask = t_ends_used <= (t_last_valid + 0.25)
    else:
        t_last_valid = None
        board_mask = np.zeros(t_ends_used.shape, dtype=bool)

    def _stat(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    idx = np.arange(trans_n.size)
    if board_mask.any():
        ax.plot(idx[board_mask], trans_n[board_mask], lw=1, color="#1f77b4", label="PnP-visible")
    if (~board_mask).any():
        ax.plot(idx[~board_mask], trans_n[~board_mask], lw=1, color="#d9480f", label="no PnP")
    ax.set_title("per-window ||t|| (mm)")
    ax.set_xlabel("window")
    ax.legend(loc="upper right", fontsize=8)
    ax = fig.add_subplot(1, 2, 2)
    if board_mask.any():
        ax.plot(pos[:-1][board_mask, 0], pos[:-1][board_mask, 1], lw=1, color="#1f77b4", label="PnP-visible")
    if (~board_mask).any():
        ax.plot(pos[:-1][~board_mask, 0], pos[:-1][~board_mask, 1], lw=1, color="#d9480f", label="no PnP")
    ax.set_title("chained xy (mm)")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "weakB_traj.png", dpi=160)
    plt.close(fig)

    summary = {
        "session_dir": str(session_dir),
        "n_frames": n_frames,
        "N_windows": int(X.shape[0]),
        "n_pnp_valid": n_pnp_any,
        "n_pnp_12x9": n_pnp_12x9,
        "t_last_pnp_valid": t_last_valid,
        "n_windows_board": int(board_mask.sum()),
        "n_windows_no_board": int((~board_mask).sum()),
        "window_rot_deg": _stat(rot_deg),
        "window_trans_mm": _stat(trans_n),
        "window_rot_deg_board": _stat(rot_deg[board_mask]),
        "window_rot_deg_no_board": _stat(rot_deg[~board_mask]),
        "window_trans_mm_board": _stat(trans_n[board_mask]),
        "window_trans_mm_no_board": _stat(trans_n[~board_mask]),
        "chain_span_mm": float(np.linalg.norm(pos[-1] - pos[0])),
        "note": "Qualitative IMU-only chain. No no-board metric ATE. Split at last 12x9 PnP time.",
    }
    (out_dir / "weakB_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
