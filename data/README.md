# Benchmark data and provenance

## `benchmark_datasets_and_trajectories.zip`

| Folder | Content |
|---|---|
| `01_Real_18_Segments/ground_truth/` | Reference poses (`pose_c2w.txt`) and intrinsics of the 18 continuous segments extracted from three 4K-referenced stereo-endoscope sessions |
| `01_Real_18_Segments/predictions/` | Trajectories (`*_est_c2w.txt`) of the compared methods and of the height-field tracker (`heightfield/`) on the same frames |
| `02_Simulated_92_Sequences/ground_truth/` | Exact poses, intrinsics, and metadata (including the motion type) of the 92 EndoEgoSim test sequences |
| `02_Simulated_92_Sequences/predictions/` | Trajectories of the compared methods (ORB-SLAM3 trajectories are partial) and of the height-field tracker |
| `03_C3VD_10_Sequences/predictions/` | Trajectories of the compared methods and of the height-field tracker; `sampled_frame_indices.json` lists the 64 evaluated frames of each sequence. Reference poses are part of the official C3VD release and are not redistributed. |
| `04_IMU_Camera_11_Sessions/heightfield_trajectories/` | Height-field trajectories (TUM format, pixel units) on every frame of the 11 held-out sessions, used by the fusion experiment |
| `05_Evaluated_Metrics/` | Per-sequence and summary metrics of the visual benchmarks, the fusion and session-level statistics, the controlled fusion study with simulated inertial increments, the learned inertial encoders, the runtime, and the transfer checks |
| `06_FixedTest_Checkerboard_Contour_Logs/` | Logs of the 11-session fixed test, the GRU6D validation and test splits, the checkerboard sequences, pose-graph drift, and the contour benchmarks |

Notes.
- The reference trajectory of the real segments combines the positions of a board observed by a fixed 4K camera with the rig IMU's orientation. Against the individual board observations it deviates by 0.33--3.56 mm in position and 1.2--5.1 deg in rotation, so position-based metrics are emphasized and no inertial rotation is evaluated against it.
- At most 64 frames per sequence are sampled uniformly, and every method is scored on the same frames. DUSt3R, MASt3R, and Endo3R were run on 32 frames because of memory limits and are scored on the 32 uniformly sampled frames.
- Sim(3) ATE is undefined when the reference camera centre does not move; the 21 simulated pure-rotation sequences are therefore scored by rotation error only.
- The simulator provides no inertial data. The controlled fusion study (`experiments/eval_controlled_fusion_sim.py`) synthesizes inertial increments from the reference poses with known noise; all other inertial results come from the real camera--IMU recordings.
- Compared-method trajectories come from the authors' evaluation harness.
- The inertial and fused metrics in `05_Evaluated_Metrics/` and `06_FixedTest_Checkerboard_Contour_Logs/` predate the correction of the camera--IMU time alignment and are superseded by the files listed below under "Time-aligned inertial evaluation".

## `results/`

- `orb_vo_fixedtest_metrics.json`: ORB-VO relative-pose metrics on the 11 held-out sessions (N = 2296).
- `vio_orbslam3_mono_fixedtest_metrics.json`: monocular ORB-SLAM3 coverage and metrics.
- `chessboard_multi_sequence_aggregate.csv`: 100-frame solvePnP comparison on five checkerboard sequences.
- `ours_posegraph_drift.csv`: open-loop vs. pose-graph ATE.
- `test5_leave_one_out_ablation.csv`: four-variant component ablation.
- `cholecseg8k_contour_scores.csv`, `cholecseg8k_fill_scores.csv`: CholecSeg8k gradient-quantile sensitivity (N = 8080).

Run `python reproduce_all.py` from the repository root to print the reported numbers.

## Time-aligned inertial evaluation (`experiments/`)

- `gyro_time_alignment_results.json`: camera--IMU latency and extrinsic rotation estimated on the 49 training and validation sessions, and the gyroscope and quaternion baselines with and without the correction.
- `inertial_classical_aligned_results.json`, `inertial_classical_aligned_errors.npz`: classical inertial baselines and their per-window errors.
- `inertial_baselines_aligned/`: GRU6D and the RoNIN and TLIO encoders trained on the aligned windows (three seeds): metrics, per-window errors, predictions, and validation accuracy.
- `fixedtest_aligned_results.json`: inertial, visual, and gated estimators, session-level statistics, the bound check, and the shared-subset comparison.
- `fusion_hybrid_results.json`: the inertial increment of the method (gyroscope rotation, GRU6D translation), its gated fusion with ORB-VO, ORB-SLAM3, and height-field increments, the weights selected on the 13 validation sessions, session-level statistics, the bound check, and the gate sensitivity (Tables 4, 5, and 16 of the manuscript).
- `review_addenda_results.json`: per-session means and post-hoc per-session time offsets (Table 6), session-averaged means, the shrinkage control for the inertial translation, the ungated gamma_R = 0 blends, per-window translation increases of the blends, the fused translation with other GRU6D seeds and encoders, a causal inertial scale, the label-noise floor and PnP flips, Holm-adjusted p-values, and the time-offset estimate from consecutive labeled frame pairs.
- `fusion_weights_validation_results.json`, `validation_sessions.json`, `validation_visual_trajectories/`: the same weight selection with the full GRU6D output as inertial branch, and the ORB-VO and height-field trajectories of the 13 validation sessions.
- `orbslam3_mono_inertial_aligned_metrics.json`, `orbslam3_mono_inertial_aligned_coverage.json`: mono-inertial ORB-SLAM3 with the corrected calibration.
- `vi_gate_sensitivity_inertial_scale.json`: sensitivity to the gate threshold and the blend weight with the full GRU6D output as inertial branch.
