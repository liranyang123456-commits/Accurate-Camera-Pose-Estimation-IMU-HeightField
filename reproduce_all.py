#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print the quantitative results of the manuscript from the bundled evaluation logs:
  - Table 3: session-disjoint split and accuracy of the evaluated GRU6D model
  - Tables 4-7: relative pose on the 11 held-out sessions (inertial, visual, gated estimators,
    learned inertial encoders), session-level statistics, ORB-SLAM3 coverage and shared subset
  - Table 8: camera pose error on the checkerboard sequences
  - Tables 9-11: height-field tracker and compared methods on the real, simulated, and C3VD sequences
  - Table 12: controlled fusion with simulated inertial increments
  - Tables 14, 17, 18: pose-graph drift and leave-one-out ablations
  - Table 16: gradient-quantile sensitivity on CholecSeg8k
  - Table 19: runtime and memory
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_visual_benchmarks import evaluate_visual_benchmarks
from experiments.run_drift_analysis import evaluate_drift_analysis
from experiments.run_imu_benchmarks import evaluate_imu_benchmarks
from experiments.run_test5_ablation import evaluate_test5_ablation
from experiments.print_extended_results import main as print_extended_results


def main():
    print("=" * 105)
    print("  CAMERA POSE ESTIMATION WITH IMU MEASUREMENTS AND A STRUCTURAL GRADIENT HEIGHT FIELD")
    print("  Results printed from the bundled evaluation logs")
    print("=" * 105)

    results_dir = REPO_ROOT / "data" / "results"

    print("\n>>> [1/5] Monocular ORB-SLAM3 coverage and quantile sensitivity...")
    evaluate_imu_benchmarks(results_dir)

    print("\n>>> [2/5] Checkerboard benchmarks...")
    evaluate_visual_benchmarks(results_dir / "chessboard_multi_sequence_aggregate.csv")

    print("\n>>> [3/5] Pose-graph drift...")
    evaluate_drift_analysis(results_dir / "ours_posegraph_drift.csv")

    print("\n>>> [4/5] Leave-one-out component ablation...")
    evaluate_test5_ablation(results_dir / "test5_leave_one_out_ablation.csv")

    print("\n>>> [5/5] Time-aligned inertial and fused estimators, session statistics, visual benchmarks, runtime, VIO diagnostics...")
    print_extended_results()

    print("\n" + "=" * 105)
    print("  Done.")
    print("=" * 105)


if __name__ == "__main__":
    main()
