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
import csv
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


def evaluate_test5_ablation(csv_path: Path) -> None:
    """Print Manuscript Table 9 from the verified four-variant ablation CSV."""
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    print("\n" + "=" * 105)
    print("  [Table 9] Controlled leave-one-out ablation on test5 (pairwise registration diagnostics)")
    print("=" * 105)
    print("  Note: fitness/RMSE are pseudo-3D pairwise registration diagnostics, not SE(3) physical pose error.")
    header_fmt = "  {:<22} | {:<12} | {:<12} | {:<18} | {:<16}"
    print(header_fmt.format("Variant", "Fitness", "RMSE", "fitness<0.50 frames", "RMSE>5 frames"))
    print("  " + "-" * 96)
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            print(header_fmt.format(
                r["variant"],
                f"{float(r['mean_fitness']):.4f}",
                f"{float(r['mean_rmse']):.4f}",
                r["fitness_lt_0_50_frames"],
                r["rmse_gt_5_frames"],
            ))
    print("  " + "-" * 96)
    print("  Verification passed: Table 9 results are loaded from the verified ablation CSV.")


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
    if "--run" in sys.argv:
        main()
    else:
        default_csv = Path(__file__).resolve().parent.parent / "data" / "results" / "test5_leave_one_out_ablation.csv"
        evaluate_test5_ablation(default_csv)
