# Accurate Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field

Official PyTorch and Python implementation and reproducible benchmarking suite for the paper:  
**"Accurate Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field"** (Submitted to *Pattern Recognition*, PR-D-26-09325).

---

## 📌 Overview

This repository provides the complete, reproducible source code, baseline evaluations, and benchmark logs for decoupled visual tracking and learned inertial pose regression in challenging endoscopic and weak-texture environments:

1. **Visual Structural Height-Field Front-End:**
   - Constructs an analytic, continuous 2.5D pseudo-height field $z_t(u, v)$ from percentile-truncated image gradients ($q=0.68$), bypassing volumetric neural rendering.
   - Computes multi-scale Point-to-Plane ICP alignment with dynamic motion quality gating (categorizing updates into Stationary/Moderate/Jump $\mathrm{S}/\mathrm{M}/\mathrm{J}$).
   - Integrates keyframe recovery and backend pose-graph optimization to constrain cumulative drift.

2. **Inertial Relative-Pose Regressor (`IMUCameraPoseGRU6D`):**
   - Employs a temporal GRU encoder with decoupled rotation and translation heads.
   - Projects a continuous 6D rotation representation (Zhou et al., CVPR 2019) to $\mathrm{SO}(3)$, supervised by adaptive Kendall multi-task uncertainty weighting.
   - Operates robustly under hardware limitations where classical tightly-coupled VIO fails to initialize (e.g., missing spatial offset $\mathbf{t}_{CI}=\mathbf{0}$, $\approx 6\,\mathrm{Hz}$ frame rates, unaligned timestamps).

```text
========================================================================================
                                 SYSTEM PIPELINE
========================================================================================

  [Visual Branch]
  Video Stream ---> Image Gradients ---> Percentile Truncation (q=0.68)
                         |
                         v
                  Pseudo-3D Height Field (u, v, h)
                         |
                         v
              Multi-scale Point-to-Plane ICP
                         |
                         v
              Quality Gating (S/M/J) & Recovery ---> Pose-Graph Refinement ---> Trajectory

  [Inertial Branch]
  IMU Stream (gyro/acc) ---> Temporal Windowing (0.5s) ---> GRU Encoder
                                                               |
                                     +-------------------------+-------------------------+
                                     |                                                   |
                                     v                                                   v
                          Rotation Head (6D -> SO(3))                         Translation Head (3D)
                                     |                                                   |
                                     +-------------------------+-------------------------+
                                                               |
                                                               v
                                                 Relative Camera Motion (SE(3))
========================================================================================
```

---

## 📂 Repository Layout

```text
Accurate-Camera-Pose-Estimation-IMU-HeightField/
├── README.md                      # Documentation and reproduction guide
├── LICENSE                        # MIT License
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules
├── reproduce_all.py               # Master one-click reproduction script
│
├── configs/                       # Configuration YAML files
│   ├── visual_pipeline.yaml       # Visual height field, ICP, and gating configs
│   └── imu_gru6d.yaml             # GRU6D model architecture and training configs
│
├── src/                           # Modular core source code
│   ├── visual/                    # Visual branch implementation
│   │   ├── height_field.py        # Gradient truncation and 2.5D elevation models
│   │   ├── mesh_registration.py   # Multi-scale Point-to-Plane ICP
│   │   ├── quality_gate.py        # S/M/J motion classifier and corner reprojection
│   │   ├── pose_graph.py          # Pose-graph optimization backend
│   │   └── exp_mesh_between_rt.py # Integrated visual pipeline runner
│   │
│   ├── imu/                       # Inertial branch implementation
│   │   ├── models.py              # IMUCameraPoseGRU6D, MLP6D, TCN6D
│   │   ├── losses.py              # PoseGeodesicLoss, AdaptivePoseGeodesicLoss
│   │   └── dataset.py             # IMU window dataset and scaling
│   │
│   └── utils/                     # Shared math, metrics, and I/O utilities
│       ├── geometry.py            # 6D rotation conversions, SE(3) algebra
│       ├── metrics.py             # ATE, RPE, geodesic rotation metrics
│       └── data_io.py             # TUM trajectory and matrix readers/writers
│
├── experiments/                   # Benchmark evaluation & ablation scripts
│   ├── run_visual_benchmarks.py   # Evaluates Table 5 (Checkerboard benchmarks)
│   ├── run_drift_analysis.py      # Evaluates Table 7 (Pose-graph drift reduction)
│   ├── run_imu_benchmarks.py      # Evaluates Table 3 & 4 (ORB-SLAM3 & baselines)
│   └── run_test5_ablation.py      # Evaluates Table 9 from verified CSV; --run re-executes the four variants
│
└── data/
    ├── README.md                  # Dataset descriptions and provenance
    └── results/                   # 100% verified experimental outputs & CSV logs
        ├── chessboard_multi_sequence_aggregate.csv   # Table 5 data
        ├── ours_posegraph_drift.csv                  # Table 7 data
        ├── vio_orbslam3_mono_fixedtest_metrics.json  # Table 3/4 data
        ├── vio_orbslam3_vs_classical_subset.json     # Table 4 tracked subset data
        └── test5_leave_one_out_ablation.csv          # Table 9 data
```

---

## ⚙️ Environment Setup

### 1. Prerequisites
- Python $\ge$ 3.8
- CUDA $\ge$ 11.3 (optional, for accelerated PyTorch training)

### 2. Installation
```bash
git clone https://github.com/liranyang123456-commits/Accurate-Camera-Pose-Estimation-IMU-HeightField.git
cd Accurate-Camera-Pose-Estimation-IMU-HeightField

# Create a virtual environment (optional)
conda create -n pose_heightfield python=3.9 -y
conda activate pose_heightfield

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 One-Click Reproduction Guide (一键复现指南)

To verify and reproduce all quantitative tables and benchmark results reported in the manuscript, run the master script:

```bash
python reproduce_all.py
```

### Reproducing Individual Tables:

#### 1. Table 5: Camera Pose Error Across Checkerboard Benchmarks
Evaluates the proposed visual height-field pipeline against DetectorFreeSfM, Reloc3r, SIFT, AKAZE, and ORB on the standardized 100-frame protocol against OpenCV `solvePnP` ground truth:
```bash
python experiments/run_visual_benchmarks.py
```
*Expected Output:*
- Proposed Visual Pipeline: Mean Rotation Error $1.622^\circ \pm 0.888^\circ$ (Median $1.062^\circ$, $72.9\%$ within $2^\circ$).
- DetectorFreeSfM: $0.908^\circ \pm 0.934^\circ$ ($96.4\%$ within $2^\circ$, ATE RMSE $12.30\,\mathrm{mm}$).
- Reloc3r: $1.249^\circ \pm 0.971^\circ$ ($81.4\%$ within $2^\circ$, ATE RMSE $46.26\,\mathrm{mm}$).

#### 2. Table 7: Backend Pose-Graph Drift Mitigation
Evaluates open-loop tracking drift vs. backend pose-graph optimization across 5 test sequences:
```bash
python experiments/run_drift_analysis.py
```
*Expected Output:*
- Open-Loop Pooled ATE: $104.16\,\mathrm{mm}$ $\rightarrow$ Pose-Graph Refined ATE: $98.38\,\mathrm{mm}$ (average reduction of $-5.5\%$).

#### 3. Table 3 & Table 4: Monocular ORB-SLAM3 and Baselines
Evaluates official ORB-SLAM3 monocular on successfully tracked test windows ($N=419$) and compares against OpenCV VO and complementary filtering on the identical subset:
```bash
python experiments/run_imu_benchmarks.py
```

#### 4. Table 9: Controlled Visual Component Leave-One-Out Ablation
Loads the verified four-variant test5 ablation metrics (full pipeline; without pose-graph; without quality gate/recovery; without multi-scale ICP):
```bash
python experiments/run_test5_ablation.py
```
*Expected Output:*
- Removing multi-scale ICP drops mean pairwise fitness from $0.5869$ to $0.4504$ and increases low-support frames (fitness $<0.50$) from $79$ to $332$.
- Removing pose-graph optimization or quality gating/recovery leaves pairwise front-end diagnostics within run-to-run variation; their roles are quantified by global drift (Table 7) and the S/M/J gating distribution.

To re-execute the four variants on the original capture workstation (advanced; requires the raw `Data_IMU_Camera_Pose_5` folders and the visual pipeline script), use:
```bash
python experiments/run_test5_ablation.py --run
```

---

## 📖 Citation

If you find this work or code useful in your research, please cite:

```bibtex
@article{li2026accurate,
  title={Accurate Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field},
  author={Li, Ranyang and Pan, Junjun and co-authors},
  journal={Pattern Recognition},
  year={2026},
  note={Under review}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
