#!/usr/bin/env python3
"""Extract the IMU stream and the motion-capture reference from TUM-VI calibrated ROS1 bags (no images).

In the calibrated release the reference topic /vrpn_client/raw_transform is the pose of the IMU in the mocap frame and is
time-aligned with the IMU (T_imu_marker = identity, zero mocap offset). Run in an environment with the rosbags package:
  python3 extract_tumvi_imu_gt.py --bag dataset-room2_512_16.bag --out <dir>/dataset-room2_512
Writes <out>/mav0/imu0/data.csv (ns, gyro rad/s, acc m/s^2) and <out>/mav0/state_groundtruth_estimate0/data.csv
(ns, position m, quaternion w x y z).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rosbags.highlevel import AnyReader


def ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out) / "mav0"
    (out / "imu0").mkdir(parents=True, exist_ok=True)
    (out / "state_groundtruth_estimate0").mkdir(parents=True, exist_ok=True)
    n_imu = n_gt = 0
    with AnyReader([Path(args.bag)]) as reader, \
            open(out / "imu0" / "data.csv", "w", newline="", encoding="utf-8") as fi, \
            open(out / "state_groundtruth_estimate0" / "data.csv", "w", newline="", encoding="utf-8") as fg:
        wi, wg = csv.writer(fi), csv.writer(fg)
        wi.writerow(["#timestamp [ns]", "w_x [rad s^-1]", "w_y", "w_z", "a_x [m s^-2]", "a_y", "a_z"])
        wg.writerow(["#timestamp [ns]", "p_x [m]", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"])
        conns = [c for c in reader.connections if c.topic in ("/imu0", "/vrpn_client/raw_transform")]
        for connection, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, connection.msgtype)
            if connection.topic == "/imu0":
                w, a = msg.angular_velocity, msg.linear_acceleration
                wi.writerow([ns(msg.header), f"{w.x:.10f}", f"{w.y:.10f}", f"{w.z:.10f}", f"{a.x:.10f}", f"{a.y:.10f}", f"{a.z:.10f}"])
                n_imu += 1
            else:
                p, q = msg.transform.translation, msg.transform.rotation
                wg.writerow([ns(msg.header), p.x, p.y, p.z, q.w, q.x, q.y, q.z])
                n_gt += 1
    print(f"DONE {args.bag}: imu {n_imu} gt {n_gt}", flush=True)


if __name__ == "__main__":
    main()
