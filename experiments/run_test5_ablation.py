#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_test5_ablation_experiments.py

Controlled 4-variant leave-one-out visual ablation on Data_IMU_Camera_Pose_5.
Evaluates:
  1. Full proposed pipeline (multi-scale point-to-plane ICP + quality gate + pose graph)
  2. w/o Pose-graph refinement (POSE_GRAPH_OPTIMIZE=0)
  3. w/o Quality gate & recovery (EVAL_GATE_ENABLE=0, PRESET_DISABLE_RECOVERY=1)
  4. w/o Multi-scale ICP (point-to-point ICP, single scale)

Results are continuously logged and saved to ablation_test5_metrics.csv.
"""

from __future__ import annotations

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np

PYTHON_EXE = r"C:\Users\lry\.conda\envs\py3d\python.exe"
EXP_SCRIPT = r"D:\reloc3r\Exp_Mehsh_Between_RT.py"
DATA_DIR = r"D:\reloc3r\Data_IMU_Camera_Pose_5\captured_photos"
BASE_OUT = r"D:\reloc3r\Data_IMU_Camera_Pose_5"
LOG_FILE = r"D:\reloc3r\ablation_test5_execution.log"
CSV_OUT = r"D:\reloc3r\ablation_test5_metrics.csv"


def log(msg: str) -> None:
    t_str = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{t_str} {msg}\n"
    print(line, end="")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


VARIANTS = [
    {
        "name": "full_pipeline",
        "desc": "Full proposed pipeline",
        "env": {
            "INPUT_DIR": DATA_DIR,
            "OUTPUT_DIR": os.path.join(BASE_OUT, "Ablation_Full"),
            "POSE_GRAPH_OPTIMIZE": "1",
            "EVAL_GATE_ENABLE": "1",
            "PRESET_DISABLE_RECOVERY": "0",
            "ICP_METHOD": "point_to_plane",
            "VOXEL_SIZE": "2.0",
            "MAX_FRAMES": "0",
            "VIS_OPEN3D": "0",
        },
    },
    {
        "name": "wo_posegraph",
        "desc": "w/o Pose-graph refinement",
        "env": {
            "INPUT_DIR": DATA_DIR,
            "OUTPUT_DIR": os.path.join(BASE_OUT, "Ablation_wo_posegraph"),
            "POSE_GRAPH_OPTIMIZE": "0",
            "EVAL_GATE_ENABLE": "1",
            "PRESET_DISABLE_RECOVERY": "0",
            "ICP_METHOD": "point_to_plane",
            "VOXEL_SIZE": "2.0",
            "MAX_FRAMES": "0",
            "VIS_OPEN3D": "0",
        },
    },
    {
        "name": "wo_quality_gate",
        "desc": "w/o Quality gate & recovery",
        "env": {
            "INPUT_DIR": DATA_DIR,
            "OUTPUT_DIR": os.path.join(BASE_OUT, "Ablation_wo_quality_gate"),
            "POSE_GRAPH_OPTIMIZE": "1",
            "EVAL_GATE_ENABLE": "0",
            "PRESET_DISABLE_RECOVERY": "1",
            "ICP_METHOD": "point_to_plane",
            "VOXEL_SIZE": "2.0",
            "MAX_FRAMES": "0",
            "VIS_OPEN3D": "0",
        },
    },
    {
        "name": "wo_multiscale_icp",
        "desc": "w/o Multi-scale ICP",
        "env": {
            "INPUT_DIR": DATA_DIR,
            "OUTPUT_DIR": os.path.join(BASE_OUT, "Ablation_wo_multiscale_icp"),
            "POSE_GRAPH_OPTIMIZE": "1",
            "EVAL_GATE_ENABLE": "1",
            "PRESET_DISABLE_RECOVERY": "0",
            "ICP_METHOD": "point_to_point",
            "VOXEL_SIZE": "1.0",
            "MAX_FRAMES": "0",
            "VIS_OPEN3D": "0",
        },
    },
]


def run_variant(v: dict) -> None:
    name = v["name"]
    desc = v["desc"]
    out_dir = v["env"]["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    log(f"Starting variant: {name} ({desc})")
    start_time = time.time()

    current_env = os.environ.copy()
    current_env.update(v["env"])

    var_log = os.path.join(out_dir, "run_console.log")
    with open(var_log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [PYTHON_EXE, EXP_SCRIPT],
            env=current_env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=r"D:\reloc3r",
        )
        proc.wait()

    elapsed = time.time() - start_time
    log(f"Completed variant: {name} in {elapsed:.1f}s, exit code: {proc.returncode}")


def main():
    log("=== Commencing Test5 Ablation Runner ===")
    for v in VARIANTS:
        try:
            run_variant(v)
        except Exception as e:
            log(f"Error running {v['name']}: {e}")
    log("=== All Ablation Variants Completed ===")


if __name__ == "__main__":
    main()
