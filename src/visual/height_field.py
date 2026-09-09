"""
Visual Front-End: Percentile-truncated gradient pseudo-height field computation.
Transforms weak-texture endoscopic or checkerboard frames into dense geometric elevation models.
(Section 3.1, Proposition 1, and Equation 8 in manuscript)
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


def compute_gradient_magnitude(img_gray: np.ndarray, blur_ksize: int = 3) -> np.ndarray:
    """
    Compute gradient magnitude field g_t(u, v) = ||nabla I_t(u, v)||_2.
    """
    if img_gray.dtype != np.uint8:
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

    if blur_ksize > 1:
        img_gray = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), 0)

    grad_x = cv2.Scharr(img_gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(img_gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(grad_x, grad_y)
    return mag


def truncate_gradient_percentile(
    grad_mag: np.ndarray,
    percentile: float = 0.68,
    z_max: float = 144.0
) -> Tuple[np.ndarray, float]:
    """
    Truncate gradient magnitude below quantile tau_q, normalize remaining support to [0, z_max].
    
    Args:
        grad_mag: Raw gradient magnitude map
        percentile: Quantile threshold (default 0.68 for 68% cut, or 0.90, 0.95)
        z_max: Maximum elevation scale for pseudo-height
    
    Returns:
        z_field: Pseudo-height field array with shape (H, W)
        tau_val: Cutoff threshold value
    """
    non_zero = grad_mag[grad_mag > 1e-4]
    if len(non_zero) == 0:
        return np.zeros_like(grad_mag, dtype=np.float32), 0.0

    tau_val = float(np.percentile(non_zero, percentile * 100.0))
    truncated = np.where(grad_mag >= tau_val, grad_mag, 0.0)

    # Normalize active supports
    mask = truncated > 0
    z_field = np.zeros_like(grad_mag, dtype=np.float32)
    if np.any(mask):
        min_v = truncated[mask].min()
        max_v = truncated[mask].max()
        if max_v > min_v:
            z_field[mask] = (truncated[mask] - min_v) / (max_v - min_v) * z_max
        else:
            z_field[mask] = z_max

    return z_field, tau_val


def stabilize_height_distribution(
    z_raw: np.ndarray,
    mu_ref: float = 24.0,
    sigma_ref: float = 12.0,
    eps_sigma: float = 1e-4
) -> np.ndarray:
    """
    Affine intensity invariant height stabilization (Equation 8 in manuscript):
      z' = ((z^raw - mu_t) / max(sigma_t, eps_sigma)) * sigma_ref + mu_ref
    """
    mask = z_raw > 1e-3
    if not np.any(mask):
        return z_raw.copy()

    mu_t = float(np.mean(z_raw[mask]))
    sigma_t = float(np.std(z_raw[mask]))
    sigma_safe = max(sigma_t, eps_sigma)

    z_stabilized = np.zeros_like(z_raw, dtype=np.float32)
    z_stabilized[mask] = ((z_raw[mask] - mu_t) / sigma_safe) * sigma_ref + mu_ref
    z_stabilized[mask] = np.maximum(z_stabilized[mask], 0.0)

    return z_stabilized


def lift_to_pointcloud_uvh(
    z_field: np.ndarray,
    step: int = 2,
    min_z: float = 0.5
) -> np.ndarray:
    """
    Lift 2D pseudo-height field to 3D point cloud in (u, v, h) coordinates:
      p = [u, v, z(u, v)]^T
    """
    H, W = z_field.shape
    v_coords, u_coords = np.mgrid[0:H:step, 0:W:step]

    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    z_flat = z_field[v_flat, u_flat]

    valid = z_flat > min_z
    pts = np.stack([u_flat[valid], v_flat[valid], z_flat[valid]], axis=1).astype(np.float64)
    return pts
