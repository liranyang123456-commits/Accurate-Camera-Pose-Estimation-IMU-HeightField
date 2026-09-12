#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Hugging Face MITI sequence_01 into EuRoC layout for ORB-SLAM3."""
from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

T_B_C0 = np.array(
    [
        [0.0, -0.4999998466695, 0.8660254923098, 0.3550000000150],
        [-1.0, 0.0, 0.0, 0.0100848356076],
        [0.0, -0.8660254923098, -0.4999998466695, -0.0550271819752],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
T_B_C1 = np.array(
    [
        [-0.0200311565580, -0.5063367902814, 0.8621031304748, 0.3547474236981],
        [-0.9997658766151, 0.0030876691468, -0.0214163081630, 0.0041072074453],
        [0.0081819754976, -0.8623302853934, -0.5062800945823, -0.0550110557371],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

SRC = Path(r"D:\reloc3r\miti_hf\sequence_01")
OUT = Path(r"D:\reloc3r\miti_euroc\sequence_01")


def _T16(vals) -> np.ndarray:
    a = np.asarray(vals, dtype=np.float64).reshape(4, 4)
    return a


def _inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def write_yaml(path: Path, T_b_c1: np.ndarray, T_c1_c2: np.ndarray) -> None:
    def mat(T: np.ndarray) -> str:
        flat = ", ".join(f"{x:.12f}" for x in T.reshape(-1))
        return (
            "!!opencv-matrix\n"
            "  rows: 4\n"
            "  cols: 4\n"
            "  dt: f\n"
            f"  data: [{flat}]"
        )

    txt = f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"

# MITI sequence_01, HF mirror FishEyeCat/MITI
# Distortion in source yaml is listed as k1,k2,k3,p1,p2; mapped to OpenCV k1,k2,p1,p2,k3.
Camera1.fx: 785.4616914439221
Camera1.fy: 785.4616914439221
Camera1.cx: 402.45582626338603
Camera1.cy: 263.9541967515214
Camera1.k1: -0.4117540717124939
Camera1.k2: 0.23465514183044434
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: -0.09278778731822968

Camera2.fx: 785.4616914439221
Camera2.fy: 785.4616914439221
Camera2.cx: 544.1066945712324
Camera2.cy: 272.25580119977616
Camera2.k1: -0.4149281978607178
Camera2.k2: 0.2392483800649643
Camera2.p1: 0.0
Camera2.p2: 0.0
Camera2.k3: -0.09897349774837494

Camera.width: 960
Camera.height: 540
Camera.fps: 59
Camera.RGB: 1

Stereo.ThDepth: 60.0
Stereo.T_c1_c2: {mat(T_c1_c2)}

IMU.T_b_c1: {mat(T_b_c1)}

# Geometry/time from MITI calibration.yaml.
# Dataset IMU densities (0.65 rad/s/sqrt(Hz)) are not usable as ORB-SLAM3 priors;
# EuRoC-like noise is used so a working VIO row can be attempted. Documented in metrics json.
IMU.NoiseGyro: 1.7e-04
IMU.NoiseAcc: 2.0e-03
IMU.GyroWalk: 1.9393e-05
IMU.AccWalk: 3.0e-03
IMU.Frequency: 200.0

ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    path.write_text(txt, encoding="utf-8")


def convert() -> None:
    cam0 = OUT / "mav0" / "cam0" / "data"
    cam1 = OUT / "mav0" / "cam1" / "data"
    imu_dir = OUT / "mav0" / "imu0"
    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)

    T_b_c0 = T_B_C0
    T_b_c1 = T_B_C1
    T_c0_c1 = _inv(T_b_c0) @ T_b_c1  # cam1 in cam0; Stereo.T_c1_c2 with c1=left=rgb_0

    # extract zips if needed
    for zip_name, dest, prefix in [
        ("rgb_0.zip", cam0, "rgb_0/"),
        ("rgb_1.zip", cam1, "rgb_1/"),
    ]:
        n_png = len(list(dest.glob("*.png")))
        if n_png >= 2000:
            print(f"skip extract {zip_name}, already {n_png} png")
            continue
        print(f"extract {zip_name}")
        with zipfile.ZipFile(SRC / zip_name) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".png"):
                    continue
                raw = Path(info.filename).name
                with z.open(info) as src, open(dest / raw, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    # map original names -> timestamp names
    times = []
    cam0_csv = []
    cam1_csv = []
    with (SRC / "rgb.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts0 = int(float(row["ts_rgb_0 (ns)"]))
            p0 = Path(row["path_rgb_0"]).name
            p1 = Path(row["path_rgb_1"]).name
            src0 = cam0 / p0
            src1 = cam1 / p1
            dst0 = cam0 / f"{ts0}.png"
            dst1 = cam1 / f"{ts0}.png"
            if src0.is_file() and src0 != dst0:
                if dst0.exists():
                    dst0.unlink()
                src0.replace(dst0)
            if src1.is_file() and src1 != dst1:
                if dst1.exists():
                    dst1.unlink()
                src1.replace(dst1)
            if dst0.is_file() and dst1.is_file():
                times.append(ts0)
                cam0_csv.append((ts0, f"data/{ts0}.png"))
                cam1_csv.append((ts0, f"data/{ts0}.png"))

    times_path = OUT / "times.txt"
    times_path.write_text("\n".join(str(t) for t in times) + "\n", encoding="utf-8")
    for csv_path, rows in [
        (OUT / "mav0" / "cam0" / "data.csv", cam0_csv),
        (OUT / "mav0" / "cam1" / "data.csv", cam1_csv),
    ]:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            f.write("#timestamp [ns],filename\n")
            for ts, fn in rows:
                f.write(f"{ts},{fn}\n")

    imu_out = imu_dir / "data.csv"
    with (SRC / "imu_0.csv").open(newline="", encoding="utf-8") as f_in, imu_out.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        f_out.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for row in csv.DictReader(f_in):
            ts = int(float(row["ts (ns)"]))
            wx, wy, wz = row["wx (rad s^-1)"], row["wy (rad s^-1)"], row["wz (rad s^-1)"]
            ax, ay, az = row["ax (m s^-2)"], row["ay (m s^-2)"], row["az (m s^-2)"]
            f_out.write(f"{ts},{wx},{wy},{wz},{ax},{ay},{az}\n")

    # GT TUM: timestamp(s) tx ty tz qx qy qz qw
    gt_tum = OUT / "groundtruth_tum.txt"
    with (SRC / "groundtruth.csv").open(newline="", encoding="utf-8") as f_in, gt_tum.open(
        "w", encoding="utf-8"
    ) as f_out:
        f_out.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in csv.DictReader(f_in):
            ts = float(row["ts (ns)"]) * 1e-9
            f_out.write(
                f"{ts:.9f} {row['tx (m)']} {row['ty (m)']} {row['tz (m)']} "
                f"{row['qx']} {row['qy']} {row['qz']} {row['qw']}\n"
            )
    # also keep euroc-like csv for eval_vio_tumvi.load_gt (wxyz after t)
    gt_csv = OUT / "groundtruth_euroc.csv"
    with (SRC / "groundtruth.csv").open(newline="", encoding="utf-8") as f_in, gt_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        f_out.write("#timestamp [ns], p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z\n")
        for row in csv.DictReader(f_in):
            f_out.write(
                f"{row['ts (ns)']},{row['tx (m)']},{row['ty (m)']},{row['tz (m)']},"
                f"{row['qw']},{row['qx']},{row['qy']},{row['qz']}\n"
            )

    yaml_path = OUT / "MITI_stereo_inertial.yaml"
    write_yaml(yaml_path, T_b_c0, T_c0_c1)
    meta = {
        "n_times": len(times),
        "n_cam0": len(list(cam0.glob("*.png"))),
        "n_cam1": len(list(cam1.glob("*.png"))),
        "t0_ns": times[0] if times else None,
        "t1_ns": times[-1] if times else None,
        "T_b_c0": T_b_c0.tolist(),
        "T_c0_c1": T_c0_c1.tolist(),
        "baseline_m": float(np.linalg.norm(T_c0_c1[:3, 3])),
    }
    (OUT / "convert_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    convert()
