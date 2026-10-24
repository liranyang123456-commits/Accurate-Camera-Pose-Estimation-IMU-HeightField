"""
Trajectory evaluation metrics: ATE (Absolute Trajectory Error) and
RPE (Relative Pose Error) under standardized benchmarking protocols.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from src.utils.geometry import geodesic_distance_so3


def compute_rpe(
    poses_est: List[np.ndarray],
    poses_gt: List[np.ndarray],
    step: int = 1,
    degrees: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute Relative Pose Error (RPE) between successive or lagged frames.

    Args:
        poses_est: List of 4x4 SE(3) estimated poses (c2w or w2c consistently)
        poses_gt: List of 4x4 SE(3) ground truth poses
        step: Frame interval / lag (e.g. 1 for consecutive frames)
        degrees: Whether to return rotation error in degrees

    Returns:
        Dictionary with 'rot_err' and 'trans_err' arrays
    """
    n = min(len(poses_est), len(poses_gt))
    rot_errors = []
    trans_errors = []

    for i in range(n - step):
        T_est_1 = poses_est[i]
        T_est_2 = poses_est[i + step]
        T_gt_1 = poses_gt[i]
        T_gt_2 = poses_gt[i + step]

        delta_est = np.linalg.inv(T_est_1) @ T_est_2
        delta_gt = np.linalg.inv(T_gt_1) @ T_gt_2

        delta_err = np.linalg.inv(delta_gt) @ delta_est
        R_err = delta_err[:3, :3]
        t_err = delta_err[:3, 3]

        rot_deg = geodesic_distance_so3(R_err, np.eye(3), degrees=degrees)
        trans_norm = float(np.linalg.norm(t_err))

        rot_errors.append(rot_deg)
        trans_errors.append(trans_norm)

    return {
        "rot_err": np.asarray(rot_errors, dtype=np.float64),
        "trans_err": np.asarray(trans_errors, dtype=np.float64),
    }


def compute_ate_aligned(
    traj_est: np.ndarray,
    traj_gt: np.ndarray,
    with_scale: bool = False
) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE) after Umeyama alignment (SE3 or Sim3).

    Args:
        traj_est: (N, 3) estimated positions
        traj_gt: (N, 3) ground truth positions
        with_scale: If True, align with Sim(3); otherwise SE(3)

    Returns:
        Dict containing RMSE, mean, std, median of translation errors
    """
    traj_est = np.asarray(traj_est, dtype=np.float64)
    traj_gt = np.asarray(traj_gt, dtype=np.float64)
    assert traj_est.shape == traj_gt.shape, "Trajectories must have same shape"

    n = traj_est.shape[0]
    mu_est = traj_est.mean(axis=0)
    mu_gt = traj_gt.mean(axis=0)

    est_c = traj_est - mu_est
    gt_c = traj_gt - mu_gt

    var_est = np.sum(est_c ** 2) / n
    H = (gt_c.T @ est_c) / n

    U, D, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    scale = 1.0
    if with_scale and var_est > 1e-12:
        scale = float(np.sum(D) / var_est)

    t = mu_gt - scale * (R @ mu_est)

    traj_est_aligned = (scale * (R @ traj_est.T)).T + t
    diff = traj_est_aligned - traj_gt
    errors = np.linalg.norm(diff, axis=1)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean = float(np.mean(errors))
    std = float(np.std(errors))
    med = float(np.median(errors))

    return {
        "rmse": rmse,
        "mean": mean,
        "std": std,
        "median": med,
        "scale": scale,
    }


def aggregate_stats(values: Union[List[float], np.ndarray], thresholds: Tuple[float, ...] = (1.0, 2.0, 5.0)) -> Dict[str, float]:
    """
    Compute summary statistics including mean, std, median, percentiles, and success hit rates.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {}

    stats = {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }

    for th in thresholds:
        stats[f"rate_lt_{th}"] = float(np.mean(arr < th) * 100.0)

    return stats
