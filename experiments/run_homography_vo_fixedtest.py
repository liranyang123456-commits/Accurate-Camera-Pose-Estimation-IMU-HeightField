#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Homography-based monocular VO (ORB-HVO) on the camera-IMU sessions, as a planar-scene front end for the fusion law.

Same protocol as ORB-VO (run_classical_vo_vio_fixedtest.run_orb_vo_on_session): frames resized to 960 px, ORB with 2500
features, ratio test 0.75, at least 30 matches. The relative pose comes from a RANSAC homography and its decomposition
(cv2.decomposeHomographyMat) instead of the essential matrix. Solutions are filtered by the visibility of the inlier
points and disambiguated by the continuity of the plane normal; the translation is the decomposition's t/d (scaled by the
plane distance). Writes <out_root>/orb_hvo/<session>/traj_tum.txt (T_WC chain) for the 11 held-out and 13 validation
sessions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from run_classical_vo_vio_fixedtest import FramePose, _load_frame_timestamps, _load_K_dist, _write_tum  # noqa: E402

ROOT = IMUCAM_ROOT
SESS = ROOT / "Data_IMU_Camera_Pose_Sequences"
OUT_ROOT = ROOT / "vio_runs_classical" / "orb_hvo"
CFG = json.load(open(ROOT / "out_train_fixed_seq_20260310_031242" / "train_config.json", encoding="utf-8"))


def pick_solution(Rs, ts, ns, pts1n, pts2n, n_prev):
    """Visible-point filter, then plane-normal continuity (n_prev is the previous normal expressed in the current first frame)."""
    cand = list(range(len(Rs)))
    if len(cand) > 1:
        try:
            keep = cv2.filterHomographyDecompByVisibleRefpoints(Rs, ns, pts1n.reshape(-1, 1, 2), pts2n.reshape(-1, 1, 2))
            if keep is not None and len(keep) > 0:
                cand = [int(k) for k in np.asarray(keep).reshape(-1)]
        except cv2.error:
            pass
    if len(cand) == 1:
        return cand[0]
    if n_prev is None:
        return max(cand, key=lambda k: abs(float(ns[k][2, 0])))
    return max(cand, key=lambda k: float(np.dot(ns[k].reshape(3), n_prev)))


def run_hvo_on_session(session_dir: Path, resize_w: int = 960, max_frames: int = 0, ransac_px: float = 2.0):
    K_full, dist, (W0, H0) = _load_K_dist(str(session_dir))
    frames = dict(_load_frame_timestamps(str(session_dir)))
    cap = cv2.VideoCapture(str(session_dir / "video.avi"))
    scale = float(resize_w) / float(max(W0, 1))
    K = K_full.copy()
    K[:2] *= scale
    dist_s = dist.copy()
    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    poses, R_wc, t_wc = [], np.eye(3), np.zeros(3)
    prev_kp = prev_des = None
    n_prev, R_prev = None, None
    n_ok = n_fail = n_pure = 0
    frame_i = -1
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames > 0 and frame_i + 1 >= max_frames):
            break
        frame_i += 1
        ts = float(frames.get(frame_i, frame_i / 30.0))
        if bgr.shape[1] != resize_w:
            bgr = cv2.resize(bgr, (resize_w, int(round(bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if prev_des is None or des is None or len(kp) < 20:
            poses.append(FramePose(t=ts, R=R_wc.copy(), p=t_wc.copy()))
            prev_kp, prev_des = kp, des
            continue
        good = [m for m, n in (mn for mn in bf.knnMatch(prev_des, des, k=2) if len(mn) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            n_fail += 1
            poses.append(FramePose(t=ts, R=R_wc.copy(), p=t_wc.copy()))
            prev_kp, prev_des = kp, des
            continue
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good])
        # undistort to the pinhole model of K before fitting the homography
        pts1u = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, dist_s, P=K).reshape(-1, 2)
        pts2u = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, dist_s, P=K).reshape(-1, 2)
        H, mask = cv2.findHomography(pts1u, pts2u, cv2.RANSAC, ransac_px)
        if H is None or mask is None or int(mask.sum()) < 20:
            n_fail += 1
            poses.append(FramePose(t=ts, R=R_wc.copy(), p=t_wc.copy()))
            prev_kp, prev_des = kp, des
            continue
        inl = mask.reshape(-1).astype(bool)
        n_sol, Rs, ts_, ns = cv2.decomposeHomographyMat(H, K)
        if n_sol == 0:
            n_fail += 1
            poses.append(FramePose(t=ts, R=R_wc.copy(), p=t_wc.copy()))
            prev_kp, prev_des = kp, des
            continue
        pts1n = cv2.undistortPoints(pts1u[inl].reshape(-1, 1, 2), K, None).reshape(-1, 2)
        pts2n = cv2.undistortPoints(pts2u[inl].reshape(-1, 1, 2), K, None).reshape(-1, 2)
        k = pick_solution(Rs, ts_, ns, pts1n, pts2n, None if n_prev is None else (R_prev @ n_prev))
        R_rel = np.asarray(Rs[k], dtype=np.float64).reshape(3, 3)  # current <- previous
        t_rel = np.asarray(ts_[k], dtype=np.float64).reshape(3)  # t / d
        n_sel = np.asarray(ns[k], dtype=np.float64).reshape(3)
        if np.linalg.norm(t_rel) < 1e-6:
            n_pure += 1
        n_prev, R_prev = n_sel, R_rel
        R_wc_new = R_wc @ R_rel.T
        t_wc = t_wc - R_wc_new @ t_rel
        R_wc = R_wc_new
        n_ok += 1
        poses.append(FramePose(t=ts, R=R_wc.copy(), p=t_wc.copy()))
        prev_kp, prev_des = kp, des
    cap.release()
    return poses, {"frames": len(poses), "ok_pairs": n_ok, "fail": n_fail, "near_pure_rotation": n_pure}


def session_lists():
    test = [Path(s).name for s in CFG["split"]["fixed_test"]["used_test_sessions"]]
    val = json.loads((HERE / "validation_sessions.json").read_text(encoding="utf-8"))["validation"]
    return test, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", choices=("test", "validation", "all"), default="all")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", type=str, default="", help="comma-separated session names")
    ap.add_argument("--out_root", type=str, default=str(OUT_ROOT))
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)
    test, val = session_lists()
    names = {"test": test, "validation": val, "all": test + val}[args.sessions]
    if args.only:
        names = [n for n in args.only.split(",") if n]
    out_root = Path(args.out_root)
    summary = {}
    for n in names:
        t0 = time.perf_counter()
        poses, info = run_hvo_on_session(SESS / n, max_frames=args.max_frames)
        out = out_root / n / "traj_tum.txt"
        _write_tum(str(out), poses)
        info["seconds"] = round(time.perf_counter() - t0, 1)
        summary[n] = info
        print("hvo", n, info, flush=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
