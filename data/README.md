# Benchmark Data and Provenance

This directory stores the official numerical evaluation outputs and data provenance for all tables reported in the manuscript:
*Accurate Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field*.

## Directory Contents

- `results/chessboard_multi_sequence_aggregate.csv`:
  Raw evaluation statistics for the standardized 100-frame protocol across the five checkerboard benchmark sequences (`seq1`--`seq4` and `line2`). Generates **Table 5**.
- `results/ours_posegraph_drift.csv`:
  Comparative drift metrics with vs. without backend pose-graph optimization across five sequences. Generates **Table 7**.
- `results/vio_orbslam3_mono_fixedtest_metrics.json`:
  Performance metrics of official monocular ORB-SLAM3 evaluated on successfully tracked test windows ($N=419$). Generates **Table 3**.
- `results/vio_orbslam3_vs_classical_subset.json`:
  Comparative benchmark metrics for ORB-SLAM3, OpenCV VO, and complementary filter (ORB+IMU) on the identical tracked subset ($N=419$). Generates **Table 4**.

## Reproducibility Protocol

All numbers in `data/results/` are directly obtained from experimental execution logs and correspond strictly to the reported paper tables without manual adjustment or smoothing. Run `python reproduce_all.py` from the repository root to verify all tables.
