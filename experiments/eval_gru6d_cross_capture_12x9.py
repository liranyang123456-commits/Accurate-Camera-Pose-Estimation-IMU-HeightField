#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild 2026-09-11 windows using only 12x9-3mm PnP labels, then re-eval old GRU6D."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\reloc3r")
SEQ_ROOT = ROOT / r"Data_IMU_Camera_Pose_Sequences\stereo_endoscope"
BUILD_PY = ROOT / "build_fixed_seq_dataset_from_session.py"
EVAL_PY = Path(__file__).resolve().parent / "eval_gru6d_cross_capture.py"
SESSIONS = [
    "pnp_seq_20260911_011115",
    "vio_seq_20260911_010902",
    "vio_seq_20260911_011013",
]


def filter_csv(src: Path, dst: Path) -> dict:
    rows = []
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            valid = str(row.get("pose_valid", "0")).strip() in ("1", "true", "True")
            name = str(row.get("chessboard_name", "")).strip()
            if valid and name != "12x9-3mm":
                row["pose_valid"] = "0"
            rows.append(row)
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    n_valid = sum(1 for r in rows if str(r.get("pose_valid", "0")).strip() in ("1", "true", "True"))
    n_12 = sum(
        1
        for r in rows
        if str(r.get("pose_valid", "0")).strip() in ("1", "true", "True")
        and str(r.get("chessboard_name", "")).strip() == "12x9-3mm"
    )
    return {"n_rows": len(rows), "n_valid_12x9": n_valid, "n_valid_12x9_check": n_12}


def main() -> None:
    sys.path.insert(0, str(EVAL_PY.parent))
    sys.path.insert(0, str(ROOT))
    import eval_gru6d_cross_capture as ev
    from imu_camera_mlp import IMUCameraMLPTrainer

    filter_stats = {}
    built = {}
    for name in SESSIONS:
        session_dir = SEQ_ROOT / name
        src = session_dir / "frames_labeled.csv"
        dst = session_dir / "frames_labeled_12x9.csv"
        filter_stats[name] = filter_csv(src, dst)
        out_dir = session_dir / "fixed_seq_len60_win0.500s_delta_window_crossR16A_12x9"
        subprocess.check_call(
            [
                sys.executable,
                str(BUILD_PY),
                "--session_dir",
                str(session_dir),
                "--frames_csv",
                str(dst),
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
        built[name] = out_dir

    trainer = IMUCameraMLPTrainer.from_checkpoint(str(ev.CKPT))
    per_seq = {}
    Xs, ys = [], []
    for name, dp in built.items():
        X = np.load(dp / "X.npy")
        y = np.load(dp / "y.npy")
        m = ev.evaluate(trainer, X, y)
        m["X_shape"] = list(X.shape)
        per_seq[name] = m
        Xs.append(X)
        ys.append(y)
        print(name, "N", m["N"], "rot_mean", m["rot_deg_mean"], "rot_p50", m["rot_deg"]["p50"], "trans_mean", m["trans_mean"])

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    pooled = ev.evaluate(trainer, X_all, y_all)
    # GT motion magnitude of the windows
    R = y_all[:, :9].reshape(-1, 3, 3)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    gt_rot = np.degrees(np.arccos(c))
    gt_trans = np.linalg.norm(y_all[:, 9:12], axis=1)
    out = {
        "filter": "pose_valid AND chessboard_name==12x9-3mm",
        "filter_stats": filter_stats,
        "per_sequence": per_seq,
        "pooled": pooled,
        "gt_window_motion": {"rot_deg": ev._stats(gt_rot), "trans": ev._stats(gt_trans)},
    }
    out_path = ev.OUT_DIR / "cross_capture_metrics_12x9.json"
    OUT = ev.OUT_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pooled": pooled, "gt": out["gt_window_motion"]}, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
