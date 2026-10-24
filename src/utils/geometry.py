"""
Geometry utilities for SE(3) and SO(3) transformations.
Supports continuous 6D rotation representations (Zhou et al., CVPR 2019)
and geodesic distances.
"""

from __future__ import annotations

import numpy as np
import torch


def rot6d_to_rotmat_torch(x6: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert continuous 6D rotation representation to 3x3 rotation matrices.
    Reference: Zhou et al., 'On the Continuity of Rotation Representations in Neural Networks', CVPR 2019.

    Args:
        x6: Tensor of shape (B, 6)
        eps: Small epsilon for numerical stability

    Returns:
        Tensor of shape (B, 3, 3) representing SO(3) rotation matrices
    """
    if x6.ndim != 2 or x6.shape[1] != 6:
        raise ValueError(f"Expected shape (B, 6), got {tuple(x6.shape)}")

    a1 = x6[:, 0:3]
    a2 = x6[:, 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=1, eps=float(eps))
    dot = (b1 * a2).sum(dim=1, keepdim=True)
    a2_ortho = a2 - dot * b1
    b2 = torch.nn.functional.normalize(a2_ortho, dim=1, eps=float(eps))
    b3 = torch.cross(b1, b2, dim=1)

    R = torch.stack([b1, b2, b3], dim=2)  # Column vectors
    return R


def rot6d_to_rotmat_np(x6: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """
    Numpy version of continuous 6D rotation to 3x3 rotation matrix.
    """
    x6 = np.asarray(x6, dtype=np.float64)
    if x6.ndim == 1:
        x6 = x6[None, :]

    a1 = x6[:, 0:3]
    a2 = x6[:, 3:6]

    norm1 = np.linalg.norm(a1, axis=1, keepdims=True)
    b1 = a1 / np.maximum(norm1, eps)

    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    a2_ortho = a2 - dot * b1
    norm2 = np.linalg.norm(a2_ortho, axis=1, keepdims=True)
    b2 = a2_ortho / np.maximum(norm2, eps)

    b3 = np.cross(b1, b2, axis=1)

    R = np.stack([b1, b2, b3], axis=-1)
    return R[0] if R.shape[0] == 1 else R


def geodesic_distance_so3(R_pred: np.ndarray, R_gt: np.ndarray, degrees: bool = True) -> float:
    """
    Compute geodesic distance (rotation angle) between two SO(3) matrices.
    d(R1, R2) = arccos((trace(R1 @ R2.T) - 1) / 2)
    """
    R_pred = np.asarray(R_pred, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)

    R_rel = R_pred @ R_gt.T
    tr = np.trace(R_rel)
    cos_angle = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    if degrees:
        return float(np.degrees(angle_rad))
    return float(angle_rad)


def invert_se3(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 SE(3) rigid transformation matrix.
    T = [R | t] -> T^-1 = [R^T | -R^T @ t]
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_se3(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Compose two 4x4 SE(3) transformation matrices.
    """
    return np.asarray(T1, dtype=np.float64) @ np.asarray(T2, dtype=np.float64)
