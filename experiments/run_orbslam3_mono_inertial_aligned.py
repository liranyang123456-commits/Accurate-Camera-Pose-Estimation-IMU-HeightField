#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run official ORB-SLAM3 mono-inertial on the 11 held-out sessions with the corrected camera-IMU calibration.

Changes against the earlier run: the IMU timestamps are shifted into the camera time base by -delta (delta from
eval_gyro_time_alignment.py, estimated on the 49 training and validation sessions), and IMU.T_b_c1 uses the R_CI of
the same calibration; the lever arm stays zero. Images, camera intrinsics, noise parameters, and the ORB settings
are those of the earlier run. Writes the IMU files and the bash driver; the driver is executed in WSL.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from export_session_to_euroc import _read_csv_dicts, _write_imu_data_csv  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = IMUCAM_ROOT
EXPORT = ROOT / "export_euroc_all"
OUT = ROOT / "export_euroc_aligned"
CFG = json.load(open(ROOT / "out_train_fixed_seq_20260310_031242" / "train_config.json", encoding="utf-8"))
WSL_CFG = "/root/orbslam3_ws/our_euroc_configs"


def main():
    al = json.loads((HERE / "gyro_time_alignment_results.json").read_text(encoding="utf-8"))
    g = al["variants"]["gyro_corrected_convention_offset"]
    delta, R_CI = float(g["delta_s"]), np.asarray(g["R_CI"])
    OUT.mkdir(exist_ok=True)
    names = [Path(s).name for s in CFG["split"]["fixed_test"]["used_test_sessions"]]
    rates = []
    for n in names:
        cam = [ln.split(",")[0] for ln in open(EXPORT / n / "mav0" / "cam0" / "data.csv", encoding="utf-8") if not ln.startswith("#")]
        t0, t1 = int(cam[0]) * 1e-9, int(cam[-1]) * 1e-9
        rows = _read_csv_dicts(str(ROOT / "Data_IMU_Camera_Pose_Sequences" / n / "imu_stream.csv"))
        (OUT / n).mkdir(exist_ok=True)
        info = _write_imu_data_csv(imu_rows=rows, out_imu_csv=str(OUT / n / "imu0_data.csv"), time_zero=False, t0_s=0.0,
                                   trim_to_cam=True, cam_t_min_s=t0 - 1.0, cam_t_max_s=t1 + 0.5, imu_time_offset_s=-delta)
        rates.append(info["imu_rate_hz_est"])
        print(n, info)
    old = subprocess.run(["wsl", "-d", "Ubuntu", "--", "cat", f"{WSL_CFG}/OurCamera_mono_inertial.yaml"],
                         capture_output=True, text=True, check=True).stdout
    T_bc = np.eye(4)
    T_bc[:3, :3] = R_CI.T
    rows = ",\n          ".join(", ".join(f"{T_bc[i, j]:.8f}" for j in range(4)) for i in range(4))
    new = re.sub(r"(IMU\.T_b_c1:.*?data: \[)(.*?)(\])", lambda m: m.group(1) + rows + m.group(3), old, flags=re.S)
    new = re.sub(r"IMU\.Frequency: .*", f"IMU.Frequency: {float(np.median(rates)):.1f}", new)
    assert new != old
    (OUT / "OurCamera_mono_inertial_aligned.yaml").write_text(new, encoding="utf-8", newline="\n")
    lines = ["#!/usr/bin/env bash", "set -uo pipefail", "export DISPLAY=",
             "ORB=/root/orbslam3_ws/ORB_SLAM3", "YAML=/mnt/d/reloc3r/export_euroc_aligned/OurCamera_mono_inertial_aligned.yaml",
             "OUTR=/mnt/d/reloc3r/vio_runs/ORB-SLAM3-mono-inertial-aligned", "mkdir -p $OUTR"]
    for n in names:
        lines += [f"S=/root/aligned_euroc/{n}; mkdir -p $S/mav0/imu0; ln -sfn /mnt/d/reloc3r/export_euroc_all/{n}/mav0/cam0 $S/mav0/cam0",
                  f"cp /mnt/d/reloc3r/export_euroc_aligned/{n}/imu0_data.csv $S/mav0/imu0/data.csv",
                  f"D=$OUTR/{n}; mkdir -p $D; cd $D",
                  f"timeout 1200 xvfb-run -a $ORB/Examples/Monocular-Inertial/mono_inertial_euroc $ORB/Vocabulary/ORBvoc.txt $YAML $S "
                  f"{WSL_CFG}/timestamps/{n}.txt > run.log 2>&1; echo exit=$? >> run.log",
                  "if [ -s CameraTrajectory.txt ]; then awk '{printf \"%.9f %s %s %s %s %s %s %s\\n\", $1/1e9,$2,$3,$4,$5,$6,$7,$8}' "
                  "CameraTrajectory.txt > traj_tum.txt; fi",
                  f"echo \"[{n}] $(tail -n 1 run.log) poses=$( [ -f traj_tum.txt ] && wc -l < traj_tum.txt || echo 0)\""]
    (OUT / "run_mono_inertial_aligned.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("delta", delta, "IMU rate", float(np.median(rates)))
    print("T_b_c1 =\n", np.round(T_bc, 4))


if __name__ == "__main__":
    main()
