# Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field

Code, evaluation logs, and benchmark trajectories for the manuscript
**"Camera Pose Estimation with IMU Measurements and a Structural Gradient Height Field"**
(Ranyang Li, Zhipeng Lin, Nan Wei, Wufeng Liu, Xiaodong Guo, Chao Fan).

## What the method does

Close-range camera tracking often runs where repeatable corners vanish and tightly coupled VIO cannot initialize (low frame rate, uncalibrated camera--IMU lever arm, software timestamps). The method decouples the visual and inertial constraints and certifies each visual increment before fusing it.

1. **Certified fusion.** Per 0.5 s window, a support test and a rotational consistency gate accept or discard the visual increment; accepted increments are blended with the inertial one on SO(3) x R^3, and rejected ones fall back to it. For any visual outlier the fused rotation error exceeds the inertial error by at most `gamma * tau_rot` (Proposition 3 in the paper), and the variance-optimal blend weight is given in closed form. The law accepts any visual front end; the paper evaluates ORB-VO, monocular ORB-SLAM3, and the height-field tracker.
2. **Initialization-free inertial increments (GRU6D).** A recurrent regressor with a continuous 6D rotation output maps a raw 0.5 s IMU window to a camera-frame SE(3) increment, without preintegration or lever-arm calibration. RoNIN and TLIO encoders trained in the same formulation are included as baselines.
3. **Training-free structural gradient height field.** Percentile-truncated Scharr gradients define a 2.5D height field that is invariant to affine photometric changes; consecutive fields are registered by multi-scale point-to-plane ICP. The paper characterizes when the registration is degenerate and which camera motions the lifted surface can represent (a pure pan or tilt cannot be represented).

## Reported results (printed by `python reproduce_all.py`)

| Benchmark | Result |
|---|---|
| 11 held-out sessions, N = 2312 windows of 0.5 s | GRU6D 2.135 deg / 9.745 mm at 100% coverage (gyroscope integration 2.785 deg; ORB-VO 5.027 deg). Gated ORB-VO blend 2.000 deg / 8.58 mm (inertial scale), improving 9 of 11 sessions in rotation and 10 of 11 in translation. Gated height-field blend 2.125 deg / 9.11 mm: rotation not significantly changed (p = 0.083), translation improved in 10 of 11 sessions (p = 0.002). Monocular ORB-SLAM3 covers 18.1% of the windows. |
| Learned inertial encoders, same heads, loss, split, and schedule (3 seeds, 120 epochs) | RoNIN-ResNet 2.128 +/- 0.050 deg / 9.09 mm; TLIO-ResNet 2.173 +/- 0.033 deg / 9.33 mm; RoNIN-LSTM 2.316 +/- 0.056 deg / 9.60 mm; GRU6D re-trained 2.145 +/- 0.035 deg / 9.45 mm. No encoder differs from the re-trained GRU6D in a session-level Wilcoxon test (p >= 0.12): the window-to-increment formulation, not the encoder, provides the gain over gyroscope integration. |
| 18 real 4K-referenced endoscope segments | Height-field tracker: Sim(3) ATE 14.22 mm (median 2.98) vs. 16.57 mm for a static trajectory (better on 18/18); not significantly different from VGGT, EndoDAC, CUT3R, Reloc3r, and AF-SfMLearner (p >= 0.13); Endo-FASt3R and Pi3 are more accurate. |
| 92 EndoEgoSim sequences | 71 sequences with camera translation: 16.69 mm vs. 30.54 mm static (better on 71/71); VGGT 15.73 mm (p = 0.22); CUT3R, EndoDAC, AF-SfMLearner, Reloc3r, Pi3, Endo-FASt3R, and DROID-SLAM are more accurate (7.78--13.31 mm). 21 pure-rotation sequences (scored by rotation only): 2.01 deg vs. 1.63 deg static, the failure predicted by the representability analysis. |
| 10 C3VD phantom sequences | 11.13 mm vs. 14.43 mm static (better on 10/10); all six learned baselines are more accurate (4.70--8.35 mm; significant for AF-SfMLearner, EndoDAC, Endo-FASt3R, and Pi3). |
| Controlled fusion with simulated inertial increments (92 simulated sequences, 18 real segments; 4 noise levels, 5 seeds) | With gamma = 0.5 and tau = 3 deg the rotation error never exceeds the inertial one by more than gamma * tau (largest excess 1.48 deg); without the gate single increments degrade by up to 20.1 deg. At sigma_R = 1 deg the blend lowers the rotation error from 1.60 to 1.22 deg (simulated) and 0.97 deg (real), and gamma* computed from the gated increments gives an error within 0.03 deg of that of the best fixed weight. Blending translation hurts when the inertial translation is accurate. The inertial increments are synthesized from the reference poses and used only for this study. |
| Runtime (Intel Core Ultra 9 275HX) | Tracker at 640 px, at most 8000 points: 257 ms per frame on one CPU thread, 66.9 ms with 24 threads. GRU6D: 17.3 ms per window on one thread. |
| CholecSeg8k, N = 8080 | Prompt-free filled support (q = 0.68): soft-tissue Dice 0.769, IoU 0.641. |

The 11 held-out sessions are checkerboard scenes rich in corners; they test the fusion law, and there ORB-VO increments are more useful than height-field increments. The checkerboard comparison and the ablations were produced with the original implementation of the tracker (`src/visual/exp_mesh_between_rt.py`: full-lattice mesh, landmark gate, recovery, keyframes, pose graph); all other visual results use `experiments/hf_tracker.py` with one configuration (640 px, q = 0.68, stride 2, at most 8000 points).

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
python experiments/run_hf_sessions.py                 # height-field trajectories on the 11 sessions
python experiments/eval_joint_hf_gru6d.py             # gated height-field + GRU6D and ORB-VO + GRU6D
python experiments/train_inertial_baselines.py --models ronin_resnet,ronin_lstm,tlio_resnet,gru6d --seeds 0 --epochs 120
python experiments/summarize_inertial_baselines.py    # mean +/- s.d. over seeds, session-level tests
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
