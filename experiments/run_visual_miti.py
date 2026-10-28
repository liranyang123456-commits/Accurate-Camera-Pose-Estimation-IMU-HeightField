#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual front ends on the nine curated MITI sequences (left camera, 960 x 540) at every 10th frame (5.9 Hz, the frame rate
of the primary rig): the height-field tracker (--what hf; needs open3d), ORB-VO (essential matrix) and ORB-HVO (homography)
on undistorted ORB matches (--what orb). Trajectories are written in TUM format with timestamps converted to the clock of
the full sensor data (mirror_time_alignment.json), so that they can be paired with the windows of build_miti_windows.py.
Output: <imu-camera root>/miti_visual/<method>/<sequence>/traj_tum.txt
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import IMUCAM_ROOT, MITI_ROOT  # noqa: E402
HF = MITI_ROOT / "hf"
ALIGN = json.loads((MITI_ROOT / "tum" / "mirror_time_alignment.json").read_text(encoding="utf-8"))
OUT = IMUCAM_ROOT / "miti_visual"
STRIDE = 10
K = np.array([[785.4616914439221, 0, 402.45582626338603], [0, 785.4616914439221, 263.9541967515214], [0, 0, 1.0]])
DIST = np.array([-0.4117540717124939, 0.23465514183044434, 0.0, 0.0, -0.09278778731822968])  # k1 k2 p1 p2 k3


def frames(seq: Path):
    rows = np.loadtxt(seq / "rgb.csv", delimiter=",", skiprows=1, usecols=(0,), dtype=np.int64)[::STRIDE]
    off = ALIGN[seq.name]["offset_s_sensordata_minus_mirror"]
    z = zipfile.ZipFile(seq / "rgb_0.zip")
    names = {int(n.split("_")[-1].split(".")[0]): n for n in z.namelist() if n.endswith(".png")}
    out = []
    for ts in rows:
        ms = int(round(ts / 1e6))
        if ms in names:
            out.append((ts * 1e-9 + off, names[ms]))
    return z, out


def read(z, name):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)


def write_tum(path: Path, ts, P):
    path.parent.mkdir(parents=True, exist_ok=True)
    q = Rotation.from_matrix(P[:, :3, :3]).as_quat()
    with open(path, "w", encoding="utf-8") as f:
        f.write("# timestamp(sensordata s) tx ty tz qx qy qz qw\n")
        for t, p, qq in zip(ts, P[:, :3, 3], q):
            f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {qq[0]:.8f} {qq[1]:.8f} {qq[2]:.8f} {qq[3]:.8f}\n")


def run_hf(seq):
    from hf_tracker import CFG, static_mask, track  # noqa: E402
    z, fr = frames(seq)
    imgs = [read(z, n) for _, n in fr]
    mask = static_mask(imgs[::4])
    res = track(imgs, mask, CFG)
    write_tum(OUT / "hf" / seq.name / "traj_tum.txt", [t for t, _ in fr], res["poses"])
    return {"frames": len(fr), "mean_fitness": float(res["fitness"].mean())}


def run_orb(seq):
    from run_homography_vo_fixedtest import pick_solution  # noqa: E402
    z, fr = frames(seq)
    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    P_vo, P_hvo = [np.eye(4)], [np.eye(4)]
    prev = None
    n_prev, R_prev = None, None
    stat = {"frames": len(fr), "vo_ok": 0, "hvo_ok": 0, "fail": 0}
    for i, (t, name) in enumerate(fr):
        gray = cv2.cvtColor(read(z, name), cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if prev is None or des is None or prev[1] is None or len(kp) < 20:
            if prev is not None:
                P_vo.append(P_vo[-1].copy())
                P_hvo.append(P_hvo[-1].copy())
            prev = (kp, des)
            continue
        good = [m for m, n in (mn for mn in bf.knnMatch(prev[1], des, k=2) if len(mn) == 2) if m.distance < 0.75 * n.distance]
        T_vo, T_hvo = np.eye(4), np.eye(4)
        if len(good) >= 30:
            p1 = np.float32([prev[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            p2 = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            u1 = cv2.undistortPoints(p1, K, DIST, P=K).reshape(-1, 2)
            u2 = cv2.undistortPoints(p2, K, DIST, P=K).reshape(-1, 2)
            E, mask = cv2.findEssentialMat(u1, u2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and mask is not None and int(mask.sum()) >= 20:
                _, R, tv, _ = cv2.recoverPose(E, u1, u2, K, mask=mask)
                T_vo[:3, :3], T_vo[:3, 3] = R, tv.reshape(3)
                stat["vo_ok"] += 1
            Hm, hmask = cv2.findHomography(u1, u2, cv2.RANSAC, 2.0)
            if Hm is not None and hmask is not None and int(hmask.sum()) >= 20:
                inl = hmask.reshape(-1).astype(bool)
                n_sol, Rs, ts_, ns = cv2.decomposeHomographyMat(Hm, K)
                if n_sol:
                    n1 = cv2.undistortPoints(u1[inl].reshape(-1, 1, 2), K, None).reshape(-1, 2)
                    n2 = cv2.undistortPoints(u2[inl].reshape(-1, 1, 2), K, None).reshape(-1, 2)
                    k = pick_solution(Rs, ts_, ns, n1, n2, None if n_prev is None else (R_prev @ n_prev))
                    T_hvo[:3, :3], T_hvo[:3, 3] = np.asarray(Rs[k]).reshape(3, 3), np.asarray(ts_[k]).reshape(3)
                    n_prev, R_prev = np.asarray(ns[k]).reshape(3), T_hvo[:3, :3].copy()
                    stat["hvo_ok"] += 1
        else:
            stat["fail"] += 1
        # T_* map previous -> current camera coordinates; the chain stores world <- camera
        P_vo.append(P_vo[-1] @ np.linalg.inv(T_vo))
        P_hvo.append(P_hvo[-1] @ np.linalg.inv(T_hvo))
        prev = (kp, des)
    ts = [t for t, _ in fr]
    write_tum(OUT / "orb_vo" / seq.name / "traj_tum.txt", ts, np.stack(P_vo))
    write_tum(OUT / "orb_hvo" / seq.name / "traj_tum.txt", ts, np.stack(P_hvo))
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=("hf", "orb"), required=True)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)
    summary = {}
    for seq in sorted(HF.glob("sequence_*")):
        summary[seq.name] = run_hf(seq) if args.what == "hf" else run_orb(seq)
        print(args.what, seq.name, summary[seq.name], flush=True)
    (OUT / f"summary_{args.what}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
