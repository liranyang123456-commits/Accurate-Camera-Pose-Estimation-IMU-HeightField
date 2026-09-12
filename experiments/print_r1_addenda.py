#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print R1 addenda JSON used in the response letter (not fused ATE)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(rel: str) -> dict:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    cov = _load("r13_coverage/coverage_metrics.json")
    print("R1-3 coverage (not ATE) vio_seq_20260911_010902")
    print(f"  mono {cov['n_frames_mono_matched']}/{cov['n_frames']}")
    print(f"  mono-inertial {cov['n_frames_mi_matched']}/{cov['n_frames']}")
    print(f"  IMU 0.5s windows {cov['n_imu_windows']}/{cov['n_frames']}")

    hard = _load("cholecseg8k_hard/protocol.json")
    print("R1-5 CholecSeg8k-Hard counts", hard["subset_counts"])

    cc = _load("gru6d_cross_capture/cross_capture_metrics_12x9.json")
    pooled = cc.get("pooled") or cc.get("overall") or {}
    print("R1-6 frozen GRU6D 12x9-only pooled", {k: pooled[k] for k in pooled if k in ("N", "rot_deg_mean", "trans_mean") or "mean" in k.lower() or k == "N"})

    miti = _load("miti_seq01/miti_metrics.json")
    si = miti["orbslam3_stereo_inertial"]
    st = miti["orbslam3_stereo_only"]
    print("R2-3 MITI stereo-inertial IMU init", si["imu_initialized"], "KF", si["kfs_at_shutdown"])
    print("R2-3 MITI stereo-only (NOT VIO) poses", st["n_saved_frame_poses"], "maps", st["n_atlas_maps"])

    weak = _load("gru6d_weak_b/vio_seq_20260912_104835/weakB_metrics.json")
    print("Weak-B keys", sorted(weak.keys())[:12])


if __name__ == "__main__":
    main()
