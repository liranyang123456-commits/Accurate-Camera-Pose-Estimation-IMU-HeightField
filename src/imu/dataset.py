"""
Dataset and preprocessing pipeline for IMU-to-Camera pose regression.
Handles temporal windowing, standard scaling, and train/val/test splits.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from src.imu.models import PoseScalerRT


class IMUCameraDataset(Dataset):
    """
    PyTorch Dataset for paired IMU windows and relative camera pose labels.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        imu_mean: Optional[np.ndarray] = None,
        imu_std: Optional[np.ndarray] = None,
        pose_scaler: Optional[PoseScalerRT] = None,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

        # Standardize IMU measurements
        if imu_mean is not None and imu_std is not None:
            std_safe = np.where(imu_std < 1e-8, 1.0, imu_std)
            self.X = (self.X - imu_mean) / std_safe

        # Scale translation components of poses if scaler provided
        if pose_scaler is not None:
            self.y = pose_scaler.transform(self.y)

        self.N, self.T, self.D = self.X.shape

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_i = torch.from_numpy(self.X[idx])  # (T, D)
        y_i = self.y[idx]

        # Extract 3x3 rotation matrix from first 9 elements (row-major)
        R_gt = torch.from_numpy(y_i[:9].reshape(3, 3))
        t_gt = torch.from_numpy(y_i[9:12])

        return x_i, R_gt, t_gt


def prepare_data_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.5,
    val_ratio: float = 0.3,
    seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split sequential data into train, val, and test subsets.
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    return {
        "train": (X[idx_train], y[idx_train]),
        "val": (X[idx_val], y[idx_val]),
        "test": (X[idx_test], y[idx_test]),
    }
