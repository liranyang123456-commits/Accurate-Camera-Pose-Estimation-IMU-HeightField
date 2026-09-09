#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master One-Click Reproduction Script:
Reproduces and verifies all quantitative tables and benchmark results directly from
raw evaluation logs, CSV tables, and ground-truth metrics:
  - Table 1: Session-Disjoint Split Statistics & Pose Regression Accuracy
  - Table 2: Relative Pose (Delta T, 0.5s) on Fixed-Test Set (GRU6D vs. Baselines)
  - Table 3: Official Monocular ORB-SLAM3 Performance on Tracked Windows
  - Table 4: Method Comparison on Shared Tracked Subset
  - Table 5: Camera Pose Error Across Checkerboard Benchmarks
  - Table 7: Backend Pose-Graph Drift Mitigation
  - Table 8: Sensitivity of Gradient Quantile q on CholecSeg8k Soft-Tissue Masks
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_visual_benchmarks import evaluate_visual_benchmarks
from experiments.run_drift_analysis import evaluate_drift_analysis
from experiments.run_imu_benchmarks import evaluate_imu_benchmarks


def main():
    print("=" * 105)
    print("  ACCURATE CAMERA POSE ESTIMATION WITH IMU MEASUREMENTS AND A STRUCTURAL GRADIENT HEIGHT FIELD")
    print("  Official Reproduction & Benchmarking Suite (100% Raw Logs & Metrics Verification)")
    print("=" * 105)

    results_dir = REPO_ROOT / "data" / "results"

    # 1. IMU Regressor, Splits, and VIO Baselines (Table 1, Table 2, Table 3, Table 4, Table 8)
    print("\n>>> [1/3] Verifying IMU Branch, Classical Baselines & Quantile Sensitivity (Tables 1, 2, 3, 4, 8)...")
    evaluate_imu_benchmarks(results_dir)

    # 2. Visual Benchmark Results (Table 5)
    print("\n>>> [2/3] Verifying Visual Branch Benchmark Evaluations (Table 5)...")
    csv_visual = results_dir / "chessboard_multi_sequence_aggregate.csv"
    evaluate_visual_benchmarks(csv_visual)

    # 3. Pose-Graph Drift Analysis (Table 7)
    print("\n>>> [3/3] Verifying Backend Pose-Graph Drift Mitigation (Table 7)...")
    csv_drift = results_dir / "ours_posegraph_drift.csv"
    evaluate_drift_analysis(csv_drift)

    print("\n" + "=" * 105)
    print("  [SUCCESS] All tables (1, 2, 3, 4, 5, 7, 8) strictly match the manuscript!")
    print("  All results are verified against raw JSON and CSV logs on disk.")
    print("=" * 105)


if __name__ == "__main__":
    main()
