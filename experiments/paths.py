#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data roots for the scripts that need raw recordings, the GRU6D training run, or the simulator harness.

IMUCAM_ROOT   directory with the IMU--camera recordings (Data_IMU_Camera_Pose_Sequences/), the evaluated
              GRU6D run (out_train_fixed_seq_20260310_031242/), the ORB-VO and ORB-SLAM3 runs
              (vio_runs_classical/, vio_runs/), and the GRU6D training and evaluation modules.
ENDOSIM_ROOT  EndoEgoSim harness (https://github.com/liranyang123456-commits/EndoEgoSim) with the test lists
              (lists/), the sequences (sim_data/), and the compared-method trajectories (results/sota/).
MITI_ROOT     MITI release (Hartwig et al., 2022): tum/sensordata.zip and tum/mirror_time_alignment.json (this
              repository's experiments/miti/ holds a copy of the alignment file) and hf/sequence_0k/ with the nine
              curated sequences of the public mirror.
All default to sibling directories of this repository and can be set as environment variables.
"""
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
IMUCAM_ROOT = Path(os.environ.get("IMUCAM_ROOT", REPO.parent / "imu_camera"))
ENDOSIM_ROOT = Path(os.environ.get("ENDOSIM_ROOT", REPO.parent / "EndoEgoSim"))
MITI_ROOT = Path(os.environ.get("MITI_ROOT", REPO.parent / "MITI"))


def eval_with_scale_correction(est, gt):
    """Sim(3) ATE, and RPE at gaps 1/5/10 after applying the Sim(3) scale (scale 1 when it is undefined)."""
    from endosim.eval.metrics import evaluate_trajectory, rpe
    res = evaluate_trajectory(est, gt)
    s = res["ate_sim3"]["scale"]
    if not np.isfinite(s) or s <= 0:
        s = 1.0
        res["ate_sim3"]["scale"] = 1.0
    est_sc = est.copy()
    est_sc[:, :3, 3] *= s
    for g in (1, 5, 10):
        if f"rpe_{g}" in res:
            res[f"rpe_{g}"] = rpe(est_sc, gt, g)
    return res
