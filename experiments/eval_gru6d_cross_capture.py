#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1-6 A: evaluate the March 2026 GRU6D checkpoint on 2026-09-11 PnP sequences.

Does not retrain. Uses:
  1) offline chessboard PnP (OpenCV, GP050-3 11x8 / 3 mm)
  2) the same fixed-window builder as training (T=60, 0.5 s, delta_window)
  3) IMU/pose scalers stored inside best_model.pth
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(r"D:\reloc3r")
CKPT = ROOT / r"out_train_fixed_seq_20260310_031242\best_model.pth"
CALIB = (
    ROOT
    / r"Data_IMU_Camera_Pose_Sequences\stereo_endoscope\calib_intrinsics_20260911_005642\camera_calibration_cam0.json"
)
SEQ_ROOT = ROOT / r"Data_IMU_Camera_Pose_Sequences\stereo_endoscope"
OUT_DIR = Path(__file__).resolve().parent / "gru6d_cross_capture"
LABEL_PY = ROOT / "label_session_chessboard_pose_offline.py"
BUILD_PY = ROOT / "build_fixed_seq_dataset_from_session.py"

SESSIONS = [
    "pnp_seq_20260911_011115",
    "vio_seq_20260911_010902",
    "vio_seq_20260911_011013",
]


class NumpyPoseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32)[:, :12])

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def label_one(session_dir: Path) -> Path:
    out_csv = session_dir / "frames_labeled.csv"
    report = session_dir / "offline_label_report.json"
    if out_csv.is_file() and report.is_file():
        print(f"skip label (exists): {out_csv}")
        return out_csv
    video = session_dir / "cam0" / "video.avi"
    if not video.is_file():
        raise FileNotFoundError(video)
    _run(
        [
            sys.executable,
            str(LABEL_PY),
            "--session_dir",
            str(session_dir),
            "--video_path",
            str(video),
            "--camera_calib_json",
            str(CALIB),
            "--out_frames_csv",
            str(out_csv),
            "--downscale",
            "0.5",
            "--max_reproj_rmse",
            "2.0",
        ]
    )
    return out_csv


def build_one(session_dir: Path, frames_csv: Path) -> Path:
    out_dir = session_dir / "fixed_seq_len60_win0.500s_delta_window_crossR16A"
    if (out_dir / "X.npy").is_file() and (out_dir / "y.npy").is_file():
        print(f"skip build (exists): {out_dir}")
        return out_dir
    _run(
        [
            sys.executable,
            str(BUILD_PY),
            "--session_dir",
            str(session_dir),
            "--frames_csv",
            str(frames_csv),
            "--out_dir",
            str(out_dir),
            "--seq_len",
            "60",
            "--window_s",
            "0.5",
            "--target_mode",
            "delta_window",
            "--start_pose_mode",
            "interp",
            "--max_reproj_rmse",
            "2.0",
            "--max_gap_s",
            "0.1",
            "--max_delta_rot_deg",
            "120",
        ]
    )
    return out_dir


def _scale_imu(scaler, X: np.ndarray) -> np.ndarray:
    n, t, d = X.shape
    xs = scaler.transform(X.reshape(-1, d)).astype(np.float32, copy=False)
    return xs.reshape(n, t, d)


def _scale_pose(scaler, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).copy()
    if hasattr(scaler, "t_mean_") and getattr(scaler, "t_mean_", None) is not None:
        y[:, 9:12] = (y[:, 9:12] - np.asarray(scaler.t_mean_, dtype=np.float32)) / np.asarray(
            scaler.t_scale_, dtype=np.float32
        )
        return y
    if hasattr(scaler, "transform"):
        return np.asarray(scaler.transform(y), dtype=np.float32)
    return y


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def evaluate(trainer, X: np.ndarray, y: np.ndarray, *, batch_size: int = 64) -> dict:
    X_s = _scale_imu(trainer.scaler_imu, X)
    y_s = _scale_pose(trainer.scaler_pose, y)
    ds = NumpyPoseDataset(X_s, y_s)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    device = trainer.device
    sp = trainer.scaler_pose
    t_mean = torch.tensor(np.asarray(sp.t_mean_, dtype=np.float32), device=device)
    t_scale = torch.tensor(np.asarray(sp.t_scale_, dtype=np.float32), device=device)
    rad2deg = 180.0 / float(np.pi)
    rot_all = []
    trans_all = []
    trainer.model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = trainer.model(xb)
            pred_u = pred.clone()
            true_u = yb.clone()
            pred_u[:, 9:12] = pred_u[:, 9:12] * t_scale + t_mean
            true_u[:, 9:12] = true_u[:, 9:12] * t_scale + t_mean
            pred_R = pred_u[:, :9].reshape(-1, 3, 3)
            true_R = true_u[:, :9].reshape(-1, 3, 3)
            trc = (pred_R * true_R).sum(dim=(1, 2))
            cos_theta = torch.clamp((trc - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
            rot_deg = torch.acos(cos_theta) * rad2deg
            trans = torch.sqrt(torch.sum((pred_u[:, 9:12] - true_u[:, 9:12]) ** 2, dim=1))
            rot_all.append(rot_deg.cpu().numpy())
            trans_all.append(trans.cpu().numpy())
    rot = np.concatenate(rot_all) if rot_all else np.zeros((0,), np.float64)
    trans = np.concatenate(trans_all) if trans_all else np.zeros((0,), np.float64)
    hits = {
        "rot_deg<=1": float(np.mean(rot <= 1.0)) if rot.size else float("nan"),
        "rot_deg<=2": float(np.mean(rot <= 2.0)) if rot.size else float("nan"),
        "rot_deg<=5": float(np.mean(rot <= 5.0)) if rot.size else float("nan"),
        "rot_deg<=10": float(np.mean(rot <= 10.0)) if rot.size else float("nan"),
        "trans<=5": float(np.mean(trans <= 5.0)) if trans.size else float("nan"),
        "trans<=10": float(np.mean(trans <= 10.0)) if trans.size else float("nan"),
        "trans<=20": float(np.mean(trans <= 20.0)) if trans.size else float("nan"),
        "trans<=50": float(np.mean(trans <= 50.0)) if trans.size else float("nan"),
    }
    return {
        "N": int(rot.size),
        "rot_deg": _stats(rot),
        "trans": _stats(trans),
        "hit_rate": hits,
        "rot_deg_mean": float(np.mean(rot)) if rot.size else float("nan"),
        "trans_mean": float(np.mean(trans)) if trans.size else float("nan"),
    }


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from imu_camera_mlp import IMUCameraMLPTrainer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT.is_file():
        raise SystemExit(f"missing checkpoint: {CKPT}")
    if not CALIB.is_file():
        raise SystemExit(f"missing calib: {CALIB}")

    label_reports = {}
    built = {}
    for name in SESSIONS:
        session_dir = SEQ_ROOT / name
        frames_csv = label_one(session_dir)
        report_path = session_dir / "offline_label_report.json"
        if report_path.is_file():
            label_reports[name] = json.loads(report_path.read_text(encoding="utf-8"))
        built[name] = str(build_one(session_dir, frames_csv))

    trainer = IMUCameraMLPTrainer.from_checkpoint(str(CKPT))
    print("device", trainer.device, "scaler_imu", type(trainer.scaler_imu), "scaler_pose", type(trainer.scaler_pose))

    per_seq = {}
    Xs, ys = [], []
    for name, d in built.items():
        dp = Path(d)
        X = np.load(dp / "X.npy")
        y = np.load(dp / "y.npy")
        m = evaluate(trainer, X, y)
        meta_path = dp / "dataset_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        m["X_shape"] = list(X.shape)
        m["y_shape"] = list(y.shape)
        m["build_stats"] = meta.get("build_stats", {})
        m["stream_stats"] = meta.get("stream_stats", {})
        per_seq[name] = m
        Xs.append(X)
        ys.append(y)
        print(name, "N", m["N"], "rot", m["rot_deg_mean"], "trans", m["trans_mean"])

    X_all = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, 60, 21), np.float32)
    y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0, 12), np.float32)
    pooled = evaluate(trainer, X_all, y_all) if X_all.shape[0] else {"N": 0}
    out = {
        "checkpoint": str(CKPT),
        "calib": str(CALIB),
        "protocol": {
            "seq_len": 60,
            "window_s": 0.5,
            "target_mode": "delta_window",
            "start_pose_mode": "interp",
            "retrained": False,
            "note": "Old GRU6D + old scalers on new 2026-09-11 PnP windows. Still checkerboard GT.",
        },
        "label_reports": label_reports,
        "per_sequence": per_seq,
        "pooled": pooled,
        "old_test_reference": {
            "N": 2312,
            "rot_deg_mean": 2.135,
            "trans_mean_mm": 9.745,
            "hit_rot_le5": 0.960,
        },
    }
    out_path = OUT_DIR / "cross_capture_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pooled": pooled, "label": {k: v.get("pose_valid_ratio") for k, v in label_reports.items()}}, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
