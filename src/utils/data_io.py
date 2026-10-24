"""
File I/O utilities for camera poses, trajectories, and sensor streams.
Supports TUM format and 4x4 matrix sequences.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np


def read_tum_trajectory(filepath: Union[str, Path]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Read TUM trajectory file format: timestamp tx ty tz qx qy qz qw.

    Returns:
        timestamps: (N,) array
        poses: List of 4x4 SE(3) matrices
    """
    timestamps = []
    poses = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue

            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])

            # Convert quaternion to rotation matrix
            # q = [qx, qy, qz, qw]
            norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if norm > 1e-9:
                qx /= norm
                qy /= norm
                qz /= norm
                qw /= norm

            R = np.array([
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
            ], dtype=np.float64)

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]

            timestamps.append(t)
            poses.append(T)

    return np.asarray(timestamps, dtype=np.float64), poses


def save_tum_trajectory(filepath: Union[str, Path], timestamps: np.ndarray, poses: List[np.ndarray]) -> None:
    """
    Save poses in TUM format: timestamp tx ty tz qx qy qz qw.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for t, T in zip(timestamps, poses):
            R = T[:3, :3]
            tx, ty, tz = T[:3, 3]

            # Rotation matrix to quaternion
            tr = np.trace(R)
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S

            f.write(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def load_4x4_matrix_file(filepath: Union[str, Path]) -> List[np.ndarray]:
    """
    Load a text file containing stacked 4x4 transformation matrices.
    """
    matrices = []
    current = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(x) for x in line.replace(",", " ").split()]
            if len(parts) == 4:
                current.append(parts)
                if len(current) == 4:
                    matrices.append(np.array(current, dtype=np.float64))
                    current = []
            elif len(parts) == 16:
                matrices.append(np.array(parts, dtype=np.float64).reshape((4, 4)))

    return matrices


def save_4x4_matrix_file(filepath: Union[str, Path], matrices: List[np.ndarray]) -> None:
    """
    Save a list of 4x4 transformation matrices to a text file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, mat in enumerate(matrices):
            f.write(f"# Matrix {idx}\n")
            for row in mat:
                f.write(f"{row[0]:.8e} {row[1]:.8e} {row[2]:.8e} {row[3]:.8e}\n")
            f.write("\n")
