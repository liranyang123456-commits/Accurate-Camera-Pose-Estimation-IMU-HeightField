"""
Quality Gating and Dynamic Motion Classifier:
Categorizes motion updates into Stationary/Moderate/Jump (S/M/J) states
and performs 2D corner reprojection verification.
(Section 3.1 & 3.2, Equations 5--7 in manuscript)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


def classify_motion_state(
    fitness: float,
    inlier_rmse: float,
    tau_fit_low: float = 0.50,
    tau_fit_high: float = 0.85,
    tau_rmse_high: float = 5.0
) -> str:
    """
    Classify frame registration update into S (Stationary), M (Moderate), or J (Jump).

    Returns:
        'S': Stationary / high confidence local match
        'M': Moderate motion / acceptable match
        'J': Jump / abrupt motion event triggering candidate recovery
    """
    if fitness >= tau_fit_high and inlier_rmse < tau_rmse_high * 0.7:
        return "S"
    elif fitness >= tau_fit_low and inlier_rmse <= tau_rmse_high:
        return "M"
    else:
        return "J"


def check_corner_reprojection_gate(
    lifted_pts_3d: np.ndarray,
    observed_pts_2d: np.ndarray,
    T_rel: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tau_px: float = 3.0,
    tau_p90: float = 4.0,
    tau_ok: float = 0.80
) -> Tuple[bool, Dict[str, float]]:
    """
    Evaluate 2D image reprojection reliability before accepting relative update T_{t-1 <- t}.
    (Equations 6--7 in manuscript)
    """
    if len(lifted_pts_3d) == 0 or len(observed_pts_2d) == 0:
        return False, {"p90_err": 999.0, "ok_ratio": 0.0}

    # Transform 3D points
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    pts_trans = (R @ lifted_pts_3d.T).T + t

    # Project to 2D image plane
    Z = np.maximum(pts_trans[:, 2], 1e-4)
    u_proj = (pts_trans[:, 0] * fx) / Z + cx
    v_proj = (pts_trans[:, 1] * fy) / Z + cy
    pts_proj_2d = np.stack([u_proj, v_proj], axis=1)

    errors = np.linalg.norm(pts_proj_2d - observed_pts_2d, axis=1)
    p90_err = float(np.percentile(errors, 90))
    ok_ratio = float(np.mean(errors <= tau_px))

    is_accepted = bool(p90_err <= tau_p90 and ok_ratio >= tau_ok)

    return is_accepted, {
        "p90_err": p90_err,
        "ok_ratio": ok_ratio,
        "mean_err": float(np.mean(errors))
    }
