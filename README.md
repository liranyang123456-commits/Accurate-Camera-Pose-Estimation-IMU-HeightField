# Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field

Code, evaluation logs, and benchmark trajectories for the manuscript
**"Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field"**
(Ranyang Li, Zhipeng Lin, Nan Wei, Wufeng Liu, Xiaodong Guo, Chao Fan).

## What the method does

Close-range camera tracking often runs where repeatable corners vanish and tightly coupled VIO cannot initialize (low frame rate, uncalibrated camera--IMU lever arm, software timestamps). The method decouples the visual and inertial constraints and tests each visual increment before fusing it.

1. **Gated fusion.** Per 0.5 s window, a support test and a rotational consistency gate accept or discard the visual increment; accepted increments are blended with the inertial one on SO(3) x R^3 with a rotation weight `gamma_R` and a translation weight `gamma_t`, and rejected ones fall back to it. For any visual outlier the fused rotation error exceeds the inertial error by at most `gamma_R * tau_rot` (Proposition 3 in the paper), and the variance-optimal weights are given in closed form. The law accepts any visual front end; the paper evaluates ORB-VO (essential matrix), ORB-HVO (homography, for planar scenes), monocular ORB-SLAM3, and the height-field tracker.
2. **Initialization-free, time-aligned inertial increments.** The camera--IMU latency of the software-timestamped rig (0.50 s) and the extrinsic rotation are estimated from the training sessions (`experiments/eval_gyro_time_alignment.py`). Over the aligned 0.5 s IMU window, gyroscope integration then gives the rotation and a recurrent regressor (GRU6D) the translation of a camera-frame SE(3) increment, without preintegration or lever-arm calibration. The full GRU6D output, the onboard quaternion, accelerometer integration, and RoNIN and TLIO encoders trained in the same formulation are included as baselines.
3. **Training-free structural gradient height field.** Percentile-truncated Scharr gradients define a 2.5D height field that is invariant to affine photometric changes; consecutive fields are registered by multi-scale point-to-plane ICP. The paper characterizes when the registration is degenerate and which camera motions the lifted surface can represent (a pure pan or tilt cannot be represented).

## Reported results (printed by `python reproduce_all.py`)

| Benchmark | Result |
|---|---|
| 11 held-out sessions, N = 2312 windows of 0.5 s | Estimated camera--IMU latency 0.50 s (per-session IQR -0.51 to -0.48 s). Inertial increment (gyroscope rotation, GRU6D translation) 1.080 deg / 9.146 mm at 100% coverage (zero motion 2.812 deg / 10.330 mm; GRU6D rotation 1.178 deg; ORB-VO 5.027 deg). Mono-inertial ORB-SLAM3 with the same calibration: 3 usable windows; monocular ORB-SLAM3: 18.1% of the windows. With the default gamma = 0.5 the largest per-window rotation increase of a gated blend is 1.46 deg (bound gamma_R * tau = 1.5 deg; 89.8 deg without the gate). With weights selected on the 13 validation sessions (gamma_R = 0.05; gamma_t = 0.55 for ORB-VO, 0.40 for the height field), the ORB-VO blend reaches 1.077 deg / 7.686 mm (translation better in 10 of 11 sessions, p = 0.003, Holm-adjusted 0.056 over the 23 tests of Table 4) and the height-field blend 1.081 deg / 8.305 mm (11 of 11, p = 0.001, Holm-adjusted 0.022). Against the GRU6D translation shrunk by a validation-selected factor of 0.82 (8.759 mm) the gains are 1.07 and 0.45 mm; they persist with every GRU6D seed and with the RoNIN and TLIO translations, with a causal inertial scale, and under session-averaged means (gyroscope rotation 2.00 deg, GRU6D rotation 1.56 deg, inertial translation 9.66 mm, blends 8.71 and 8.98 mm; the two largest sessions hold 64% of the windows). An ungated blend with gamma_R = 0 reaches 7.587 and 8.282 mm. With the homography front end ORB-HVO (1.851 deg / 5.882 mm alone, reference scale) the gated blend with validation weights (gamma_R = 0.05, gamma_t = 0.85) reaches 1.079 deg / 5.900 mm (-35%, 11 of 11 sessions, Holm-adjusted p = 0.022; 2.86 mm against the shrunk translation). |
| MITI laparoscopic intervention (recorded IMU, infrared optical-tracking reference; regressors retrained on 194 s of the intervention, tested on the nine curated sequences: N = 3180 windows of 0.5 s ending at every 20 Hz tracker sample, 318 without overlap) | Reference motion per window 5.44 deg / 8.11 mm. Gyroscope integration (IMU-to-tracker offset 0.167 s estimated on the training windows) 0.92 deg. Retrained GRU6D: rotation 4.00 deg, translation 8.06 mm (7.98--8.18 over three seeds; 0.6% below zero translation); RoNIN-LSTM 7.59--7.86 mm; TLIO-ResNet 6.76--7.15 mm. Visual front ends at 5.9 Hz are all worse than zero motion: height field 5.83 deg / 8.81 mm, ORB-VO 20.4 deg / 8.92 mm, ORB-HVO 11.3 deg / 12.5 mm (reference scale). Gated blends with the checkerboard weights keep 0.91--0.92 deg (largest per-window increase 0.15 deg; 1.48--1.49 deg at gamma = 0.5, bound 1.5 deg) and give 7.91--7.93 mm (GRU6D translation) or 7.11--7.17 mm (TLIO translation); ungated blends degrade to 3.0--10.3 deg. Best translation of any estimator: 6.94 mm (ungated height-field blend with the TLIO translation). At the native 59 Hz the height field chains 29 registrations per window and its rotation error on sequence 01 rises from 3.8 to 6.1 deg. |
| TUM-VI Room 1 frame-rate sweep (hardware-synchronized, Kalibr-calibrated) | ORB-SLAM3 mono-inertial initializes and tracks at 20 / 10 / 6.7 Hz with ATE 0.016 / 0.052 / 0.100 m; VINS-Fusion stereo-inertial degrades to a 0.5 s translation RPE of 0.049 / 0.140 / 2.61 m, with non-deterministic ROS 2 replay (a second 20 Hz run: ATE 0.148 m). The 6 Hz rate alone does not explain the initialization failure on the primary rig. |
| Learned inertial encoders, same heads, loss, split, and schedule (3 seeds, 120 epochs) | RoNIN-ResNet 1.143 +/- 0.040 deg / 8.64 mm; RoNIN-LSTM 1.150 +/- 0.001 deg / 8.47 mm; TLIO-ResNet 1.292 +/- 0.073 deg / 8.31 mm; GRU6D 1.194 +/- 0.019 deg / 8.98 mm. No encoder differs from GRU6D in a session-level Wilcoxon test (p >= 0.067). Without the time alignment, gyroscope integration reaches 2.33 deg and GRU6D 2.15 deg (`experiments/inertial_baselines/`). |
| 18 real 4K-referenced endoscope segments | Height-field tracker: Sim(3) ATE 14.22 mm (median 2.98) vs. 16.57 mm for a static trajectory (better on 18/18); within the 13.66--14.90 mm range of seven learned models; after Holm correction no paired difference to a learned model is significant (unadjusted p = 0.009 for Endo-FASt3r and 0.048 for Pi3). |
| 92 EndoEgoSim sequences | 71 sequences with camera translation: 16.69 mm vs. 30.54 mm static (better on 71/71); VGGT 15.73 mm (p = 0.22); CUT3R, EndoDAC, AF-SfMLearner, Reloc3r, Pi3, Endo-FASt3r, and DROID-SLAM are more accurate (7.78--13.31 mm). 21 pure-rotation sequences (scored by rotation only): 2.01 deg vs. 1.63 deg static, the failure predicted by the representability analysis. |
| 10 C3VD phantom sequences | 11.13 mm vs. 14.43 mm static (better on 10/10); all six learned baselines are more accurate (4.70--8.35 mm; after Holm correction significant for Endo-FASt3r). |
| Controlled fusion with simulated inertial increments (92 simulated sequences, 18 real segments; 4 noise levels, 5 seeds) | With gamma = 0.5 and tau = 3 deg the rotation error never exceeds the inertial one by more than gamma * tau (largest excess 1.48 deg); without the gate single increments degrade by up to 20.1 deg. At sigma_R = 1 deg the blend lowers the rotation error from 1.60 to 1.22 deg (simulated) and 0.97 deg (real), and gamma* computed from the gated increments gives an error within 0.03 deg of that of the best fixed weight. Blending translation hurts when the inertial translation is accurate. The inertial increments are synthesized from the reference poses and used only for this study. |
| Runtime (Intel Core Ultra 9 275HX) | Tracker at 640 px, at most 8000 points: 257 ms per frame on one CPU thread, 66.9 ms with 24 threads. GRU6D: 17.3 ms per window on one thread. |
| CholecSeg8k, N = 8080 | Prompt-free filled support (q = 0.68): soft-tissue Dice 0.769, IoU 0.641, below an all-foreground mask (0.813 / 0.690), so region overlap does not rank the methods on this benchmark. |

The 11 held-out sessions are checkerboard scenes rich in corners; they test the fusion law, and there ORB-VO increments are more useful than height-field increments. The checkerboard comparison and the ablations were produced with the original implementation of the tracker (`src/visual/exp_mesh_between_rt.py`: full-lattice mesh, landmark gate, recovery, keyframes, pose graph); all other visual results use `experiments/hf_tracker.py` with one configuration (640 px, q = 0.68, stride 2, at most 8000 points).

## Correction of the inertial evaluation

An earlier version of this repository evaluated the inertial branch without compensating the camera--IMU latency of the software-timestamped rig and with the classical gyroscope and quaternion baselines integrated in the direction opposite to the labels. With both corrected (Section 3.5 of the manuscript), every inertial estimate roughly halves its rotation error; the tables above and `reproduce_all.py` report the corrected evaluation. The unaligned encoder runs remain in `experiments/inertial_baselines/` for the alignment ablation.

## Repository layout

```text
README.md, LICENSE, requirements.txt, reproduce_all.py
configs/        visual_pipeline.yaml, imu_gru6d.yaml
src/            visual/ (height field, registration, gate, pose graph), imu/ (GRU6D, losses, dataset), utils/
experiments/    benchmark and evaluation scripts, result JSON/CSV files, figures/
data/           README.md, results/ (CSV and JSON logs), benchmark_datasets_and_trajectories.zip
```

## Reproduction

```bash
pip install -r requirements.txt
python reproduce_all.py
```

`reproduce_all.py` prints the manuscript tables from the bundled logs. The scripts that regenerate the logs need the raw recordings, the evaluated GRU6D run, or the EndoEgoSim harness; set `IMUCAM_ROOT` and `ENDOSIM_ROOT` as described in `experiments/paths.py`:

```bash
python experiments/eval_gyro_time_alignment.py        # camera-IMU latency and R_CI from the 49 training/validation sessions
python experiments/build_aligned_windows.py --delta -0.50   # IMU windows shifted by the latency
python experiments/eval_inertial_classical_aligned.py # gyroscope, quaternion, and accelerometer baselines
python experiments/train_inertial_baselines.py --models ronin_resnet,ronin_lstm,tlio_resnet,gru6d --seeds 0 --epochs 120 \
       --ds fixed_seq_len60_win0.500s_delta_window_offset-0.50 --out experiments/inertial_baselines_aligned
python experiments/summarize_inertial_baselines.py --dir experiments/inertial_baselines_aligned --suffix _aligned
python experiments/run_hf_sessions.py                 # height-field trajectories on the 11 sessions
python experiments/run_visual_validation_sessions.py --what orbvo   # and --what hf: visual increments on the 13 validation sessions
python experiments/eval_fixedtest_aligned.py          # inertial, visual, and gated estimators; session statistics
python experiments/eval_fusion_weights_validation.py  # blend weights selected on the validation sessions (GRU6D as inertial branch)
python experiments/run_homography_vo_fixedtest.py     # homography front end (ORB-HVO) on the 11 held-out and 13 validation sessions
python experiments/eval_fusion_hybrid.py              # inertial increment of the method and its gated fusion with ORB-VO, ORB-HVO, ORB-SLAM3, height field (Tables 4, 5, 18)
python experiments/eval_review_addenda.py             # per-session table, shrinkage control, ungated baseline, Holm adjustment, offset diagnostics (Table 8)
python experiments/eval_miti_target_regime.py --run-hf --stride 10 && python experiments/eval_miti_target_regime.py --stride 10   # MITI sequence 01 at 5.9 Hz (chaining check; --stride 1 for 59 Hz)
python experiments/build_miti_windows.py              # MITI windows from sensordata.zip + mirror_time_alignment.json (IMU1, IR tracking, offset estimate)
python experiments/train_miti_gru6d.py --models gru6d,ronin_lstm,tlio_resnet --seeds 0,1,2 --epochs 120   # regressors retrained on the intervention
python experiments/run_visual_miti.py --what hf && python experiments/run_visual_miti.py --what orb         # height field, ORB-VO, ORB-HVO on the nine sequences at 5.9 Hz
python experiments/eval_miti_fusion.py                # gated fusion on the nine MITI sequences (Table 7)
python experiments/make_fig_summary.py                # summary figure: session-level gains and error relative to the reference on every benchmark (Fig. 13; Table 23)
python experiments/run_tumvi_framerate.py --strides 1,2,3   # ORB-SLAM3 and VINS-Fusion at 20/10/6.7 Hz on TUM-VI room1 (WSL; Table 6)
python experiments/run_orbslam3_mono_inertial_aligned.py    # recalibrated mono-inertial ORB-SLAM3 inputs (run in WSL)
python experiments/run_hf_benchmarks.py               # tracker on the real, simulated, and C3VD sequences
python experiments/reeval_benchmarks.py               # one metric implementation for every method
python experiments/make_visual_benchmark_figures.py   # figures and table rows
python experiments/time_hf_tracker.py                 # single-thread latency and memory
python experiments/time_hf_variants.py                # multi-threaded latency
```

## Data

`data/benchmark_datasets_and_trajectories.zip` (reference poses and compared-method trajectories of the real segments and simulated sequences, compared-method trajectories and sampled frame indices of the C3VD sequences, height-field trajectories of every sequence and of the 11 sessions, and evaluation logs; no images) is described in `data/README.md`. The EndoEgoSim simulator and test list are at <https://github.com/liranyang123456-commits/EndoEgoSim>. Public datasets (CholecSeg8k, BSDS500, BIPEDv2, COCO 2017, C3VD, MITI, TUM-VI) remain at their official sources.

## Citation

```bibtex
@article{li2026camera,
  title={Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field},
  author={Li, Ranyang and Lin, Zhipeng and Wei, Nan and Liu, Wufeng and Guo, Xiaodong and Fan, Chao},
  year={2026},
  note={Manuscript submitted to Expert Systems with Applications}
}
```

## License

MIT License; see [LICENSE](LICENSE).
