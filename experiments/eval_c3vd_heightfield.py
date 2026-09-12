#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Height-field tracking on C3VD (visual branch only; not GRU6D, not VIO).

Supports:
  1) Registered C3VD folders: pose.txt with 16 floats/line (row-major c2w) + *_color.png
  2) Official C3VD *sample* zip/folder: rgb/NNNN.png + timed robot pose log + hand-eye,
     matching DurrLab/C3VD AlignmentModule (FPS=29.97, A2B, poseStartTime).

Pseudo-height z is not metric depth. Report RPE rotation and Sim(3) ATE.
RPE translation is not reported in millimetres.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(r"D:\reloc3r\Accurate-Camera-Pose-Estimation-IMU-HeightField")
sys.path.insert(0, str(REPO))

from src.visual.height_field import (  # noqa: E402
    compute_gradient_magnitude,
    lift_to_pointcloud_uvh,
    stabilize_height_distribution,
    truncate_gradient_percentile,
)
from src.visual.mesh_registration import register_pointclouds_multiscale  # noqa: E402
from src.utils.metrics import compute_ate_aligned, compute_rpe  # noqa: E402

C3VD_FPS = 29.97
SAMPLE_RGB_PREFIX = "sampledata/rgb/"


def _mat4_colmajor(vals) -> np.ndarray:
    return np.asarray(vals, np.float64).reshape((4, 4), order="F")


def _orthonormalize_R(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def _slerp_pose(T0: np.ndarray, T1: np.ndarray, w: float) -> np.ndarray:
    w = float(np.clip(w, 0.0, 1.0))
    R0 = Rotation.from_matrix(_orthonormalize_R(T0[:3, :3]))
    R1 = Rotation.from_matrix(_orthonormalize_R(T1[:3, :3]))
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R0, R1]))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = slerp([w]).as_matrix()[0]
    T[:3, 3] = (1.0 - w) * T0[:3, 3] + w * T1[:3, 3]
    return T


def _parse_ini_mat4(line: str) -> np.ndarray:
    rhs = line.split("=", 1)[1]
    vals = [float(x.strip()) for x in rhs.split(",") if x.strip()]
    if len(vals) != 16:
        raise ValueError(f"expected 16 values, got {len(vals)}: {line[:80]}")
    return _mat4_colmajor(vals)


def load_sample_config(text: str) -> dict:
    pose_start = None
    mats = {}
    for raw in text.splitlines():
        s = raw.split(";", 1)[0].strip()
        if not s or "=" not in s:
            continue
        key = s.split("=", 1)[0].strip()
        if key == "poseStartTime":
            pose_start = float(s.split("=", 1)[1].strip())
        elif key in ("A_cal", "B_cal", "X"):
            mats[key] = _parse_ini_mat4(s)
    if pose_start is None or any(k not in mats for k in ("A_cal", "B_cal", "X")):
        raise ValueError("config.ini missing poseStartTime or A_cal/B_cal/X")
    return {"poseStartTime": pose_start, **mats}


def a2b(A: np.ndarray, A_cal: np.ndarray, B_cal: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Camera pose B from robot pose A (DurrLab/C3VD Handeye.cpp)."""
    return B_cal @ np.linalg.inv(X) @ np.linalg.inv(A_cal) @ A @ X


def load_robot_pose_log(text: str) -> tuple[np.ndarray, np.ndarray]:
    times, poses = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        vals = [float(x) for x in s.split(",")]
        if len(vals) != 17:
            continue
        times.append(vals[0])
        poses.append(np.asarray(vals[1:], np.float64).reshape(4, 4))
    t = np.asarray(times, np.float64)
    P = np.stack(poses, 0)
    order = np.argsort(t)
    return t[order], P[order]


def interp_robot(t_query: float, times: np.ndarray, poses: np.ndarray) -> np.ndarray:
    if t_query <= times[0]:
        return poses[0].copy()
    if t_query >= times[-1]:
        return poses[-1].copy()
    i1 = int(np.searchsorted(times, t_query, side="left"))
    i0 = max(i1 - 1, 0)
    if times[i1] == times[i0]:
        return poses[i0].copy()
    w = (t_query - times[i0]) / (times[i1] - times[i0])
    return _slerp_pose(poses[i0], poses[i1], w)


def _load_registered_pose_txt(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals = [float(x) for x in s.replace(",", " ").split()]
        if len(vals) == 16:
            rows.append(np.asarray(vals, np.float64).reshape(4, 4))
        elif len(vals) == 17:
            rows.append(np.asarray(vals[1:], np.float64).reshape(4, 4))
    if not rows:
        raise ValueError(f"no 4x4 poses in {path}")
    return np.stack(rows, 0)


def _height_cloud(img_bgr: np.ndarray, mask: np.ndarray | None, q: float, step: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mag = compute_gradient_magnitude(gray)
    z, _ = truncate_gradient_percentile(mag, percentile=q)
    z = stabilize_height_distribution(z)
    if mask is not None:
        if mask.shape[:2] != z.shape:
            mask = cv2.resize(mask, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_NEAREST)
        z = np.where(mask > 127, z, 0.0)
    pts = lift_to_pointcloud_uvh(z, step=step)
    if pts.shape[0] > 8000:
        idx = np.linspace(0, pts.shape[0] - 1, 8000).astype(int)
        pts = pts[idx]
    return pts


def _decode_png(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img


def _resize_max_w(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(round(h * max_w / w))), interpolation=cv2.INTER_AREA)


def load_sample_from_zip(zip_path: Path):
    zf = zipfile.ZipFile(zip_path)
    cfg = load_sample_config(zf.read("sampledata/config.ini").decode("utf-8"))
    times, robot = load_robot_pose_log(zf.read("sampledata/pose.txt").decode("utf-8"))
    rgb_names = sorted(
        n for n in zf.namelist() if n.startswith(SAMPLE_RGB_PREFIX) and n.endswith(".png")
    )
    mask = None
    if "sampledata/mask.png" in zf.namelist():
        mask = cv2.imdecode(np.frombuffer(zf.read("sampledata/mask.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    return zf, cfg, times, robot, rgb_names, mask


def sample_camera_poses(n_rgb: int, cfg: dict, times: np.ndarray, robot: np.ndarray) -> np.ndarray:
    out = []
    for i in range(n_rgb):
        t = i / C3VD_FPS + cfg["poseStartTime"]
        A = interp_robot(t, times, robot)
        B = a2b(A, cfg["A_cal"], cfg["B_cal"], cfg["X"])
        out.append(B)
    return np.stack(out, 0)


def track_clouds(clouds: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict]]:
    T_cum = np.eye(4)
    T_list = [T_cum.copy()]
    pair_fit = []
    for i in range(1, len(clouds)):
        T_pair, fit, rmse = register_pointclouds_multiscale(clouds[i], clouds[i - 1])
        T_cum = T_cum @ T_pair
        T_list.append(T_cum.copy())
        pair_fit.append({"i": i, "fitness": float(fit), "rmse": float(rmse), "n_src": int(clouds[i].shape[0])})
        if i % 10 == 0 or i == len(clouds) - 1:
            print(f"  icp {i}/{len(clouds)-1} fitness={fit:.3f} rmse={rmse:.3f}", flush=True)
    return T_list, pair_fit


def summarize(T_list, gt, seq_name: str, extra: dict) -> dict:
    rpe = compute_rpe(T_list, gt, step=1, degrees=True)
    traj_est = np.stack([T[:3, 3] for T in T_list], 0)
    traj_gt = np.stack([T[:3, 3] for T in gt], 0)
    ate_sim3 = compute_ate_aligned(traj_est, traj_gt, with_scale=True)
    ate_se3 = compute_ate_aligned(traj_est, traj_gt, with_scale=False)
    rel_rot = np.asarray(rpe["rot_err"], np.float64)
    out = {
        "dataset": "C3VD",
        "sequence": seq_name,
        "n_frames": len(T_list),
        "note": (
            "Visual height-field tracking only. Not GRU6D. Not VIO. "
            "Silicone colon phantom, not in-vivo mucosa. No IMU. "
            "z is not metric depth; translation uses Sim(3) Umeyama. "
            "RPE translation is not millimetres."
        ),
        "rpe_rot_deg_mean": float(np.mean(rel_rot)) if rel_rot.size else None,
        "rpe_rot_deg_median": float(np.median(rel_rot)) if rel_rot.size else None,
        "rpe_rot_deg_std": float(np.std(rel_rot)) if rel_rot.size else None,
        "rpe_rot_lt1_pct": float(np.mean(rel_rot < 1.0) * 100.0) if rel_rot.size else None,
        "rpe_rot_lt2_pct": float(np.mean(rel_rot < 2.0) * 100.0) if rel_rot.size else None,
        "rpe_rot_lt5_pct": float(np.mean(rel_rot < 5.0) * 100.0) if rel_rot.size else None,
        "ate_sim3_rmse": ate_sim3.get("rmse"),
        "ate_sim3_mean": ate_sim3.get("mean"),
        "ate_sim3_median": ate_sim3.get("median"),
        "ate_sim3_scale": ate_sim3.get("scale"),
        "ate_se3_rmse": ate_se3.get("rmse"),
        **extra,
    }
    return out


def build_index(n_total: int, stride: int, max_frames: int, start: int) -> list[int]:
    idx = list(range(start, n_total, max(stride, 1)))
    if max_frames > 0:
        idx = idx[:max_frames]
    if len(idx) < 3:
        raise SystemExit(f"too few frames: n_total={n_total} stride={stride} max_frames={max_frames}")
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="", help="C3VD sample zip (rgb/ + timed pose.txt)")
    ap.add_argument("--seq_dir", default="", help="Extracted sequence folder")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=80)
    ap.add_argument("--max_width", type=int, default=640)
    ap.add_argument("--q", type=float, default=0.68)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    zf = None
    mask = None
    images_cb = None
    poses_all = None
    seq_name = ""

    if args.zip:
        zip_path = Path(args.zip)
        zf, cfg, times, robot, rgb_names, mask = load_sample_from_zip(zip_path)
        n_rgb = len(rgb_names)
        poses_all = sample_camera_poses(n_rgb, cfg, times, robot)
        seq_name = zip_path.name + f"/sampledata (FPS={C3VD_FPS}, poseStartTime={cfg['poseStartTime']})"
        print(
            f"sample zip: {n_rgb} rgb, {len(times)} robot poses, "
            f"pose t=[{times[0]:.3f},{times[-1]:.3f}]",
            flush=True,
        )

        def images_cb(i: int) -> np.ndarray:
            return _decode_png(zf.read(rgb_names[i]))

        extra_meta = {
            "source": "C3VD official sample (robot pose log + hand-eye X, not registered pose.txt)",
            "fps": C3VD_FPS,
            "poseStartTime": cfg["poseStartTime"],
            "n_rgb": n_rgb,
            "n_robot_poses": int(len(times)),
            "gt_units": "millimetres (robot/hand-eye)",
        }
    else:
        seq = Path(args.seq_dir)
        poses_all = _load_registered_pose_txt(seq / "pose.txt")
        colors = sorted(seq.glob("*_color.png"))
        if not colors:
            colors = sorted((seq / "rgb").glob("*.png")) if (seq / "rgb").is_dir() else []
        if not colors:
            colors = [p for p in sorted(seq.glob("*.png")) if "depth" not in p.name]
        n = min(len(colors), poses_all.shape[0])
        poses_all = poses_all[:n]
        seq_name = str(seq)

        def images_cb(i: int) -> np.ndarray:
            img = cv2.imread(str(colors[i]), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"unreadable {colors[i]}")
            return img

        extra_meta = {"source": "registered C3VD folder", "n_rgb": n}

    idx = build_index(len(poses_all), args.stride, args.max_frames, args.start)
    extra_meta.update({"stride": args.stride, "start": args.start, "indices": idx[:3] + ["..."] + idx[-3:] if len(idx) > 6 else idx})
    print(f"tracking {len(idx)} frames (stride={args.stride}, start={args.start})", flush=True)

    clouds = []
    for k, i in enumerate(idx):
        img = _resize_max_w(images_cb(i), args.max_width)
        m = None
        if mask is not None:
            m = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        clouds.append(_height_cloud(img, m, args.q, args.step))
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  height {k+1}/{len(idx)} pts={clouds[-1].shape[0]}", flush=True)

    T_list, pair_fit = track_clouds(clouds)
    gt = [poses_all[i] for i in idx]
    extra_meta["pair_fitness_mean"] = float(np.mean([p["fitness"] for p in pair_fit]))
    extra_meta["n_low_support"] = int(sum(1 for p in pair_fit if p["fitness"] < 0.2))
    extra_meta["pair_fitness_median"] = float(np.median([p["fitness"] for p in pair_fit]))

    out = summarize(T_list, gt, seq_name, extra_meta)
    out_path = Path(args.out_json) if args.out_json else Path(r"E:\elsarticle-templateCMBP_IMU_Camera\PR_R1_Revision\experiments\c3vd_heightfield\heightfield_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "indices"}, indent=2))
    if zf is not None:
        zf.close()


if __name__ == "__main__":
    main()
