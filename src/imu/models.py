"""
Inertial Pose Regressor Models: IMUCameraPoseGRU6D and baseline architectures.
Encodes 6-axis IMU sequences and regresses relative camera SE(3) increments
via continuous 6D rotation representation and 3D translation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.utils.geometry import rot6d_to_rotmat_torch


class PoseScalerRT:
    """
    Scaler for 12D SE(3) pose labels [R(9) row-major + t(3)].
    Rotation elements are kept unscaled; translation is normalized by mean/std.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean_t = np.zeros(3, dtype=np.float32)
        self.std_t = np.ones(3, dtype=np.float32)

    def fit(self, y: np.ndarray) -> "PoseScalerRT":
        y = np.asarray(y, dtype=np.float32)
        t = y[:, 9:12]
        self.mean_t = np.mean(t, axis=0)
        self.std_t = np.std(t, axis=0)
        self.std_t = np.where(self.std_t < self.eps, 1.0, self.std_t)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y_out = y.copy().astype(np.float32)
        y_out[:, 9:12] = (y_out[:, 9:12] - self.mean_t) / self.std_t
        return y_out

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y_out = y_scaled.copy().astype(np.float32)
        y_out[:, 9:12] = y_out[:, 9:12] * self.std_t + self.mean_t
        return y_out


class _ResidualMLPBlockLN(nn.Module):
    """Pre-norm residual MLP block: x + Dropout(Linear(Dropout(GELU(Linear(LayerNorm(x))))))."""

    def __init__(self, dim: int, *, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        h = max(1, int(dim) * int(expansion))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop2(self.fc2(self.drop1(self.act(self.fc1(self.norm(x))))))


class IMUCameraPoseGRU6D(nn.Module):
    """IMU-to-camera relative-pose regressor of the paper (Fig. 3); same layers and names as experiments/imu_camera_mlp.py,
    so the checkpoints of the evaluated runs load directly.

    Linear(input_dim -> embed_dim) + LayerNorm, a GRU (gru_num_layers x gru_hidden_dim) whose last hidden state feeds a
    shared LayerNorm-Linear-GELU layer (head_dim), then two residual MLP blocks per head before a 6D rotation output
    (Gram-Schmidt to SO(3)) and a 3D translation output. Defaults are the evaluated configuration (11.44 M parameters).
    forward() returns the 12-D pose [R row-major (9), t (3)].
    """

    def __init__(self, *, input_dim: int = 21, embed_dim: int = 256, gru_hidden_dim: int = 512, gru_num_layers: int = 2,
                 head_dim: int = 512, head_blocks: int = 2, dropout: float = 0.25):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed = nn.Linear(input_dim, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden_dim, num_layers=gru_num_layers, batch_first=True,
                          dropout=dropout if gru_num_layers > 1 else 0.0)
        self.shared = nn.Sequential(nn.LayerNorm(gru_hidden_dim), nn.Linear(gru_hidden_dim, head_dim), nn.GELU(), nn.Dropout(dropout))
        self.rot_blocks = nn.Sequential(*[_ResidualMLPBlockLN(head_dim, expansion=4, dropout=dropout) for _ in range(head_blocks)])
        self.trans_blocks = nn.Sequential(*[_ResidualMLPBlockLN(head_dim, expansion=4, dropout=dropout) for _ in range(head_blocks)])
        self.rot_head = nn.Linear(head_dim, 6)
        self.trans_head = nn.Linear(head_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) -> (B, 12) = [R row-major, t]."""
        _, h_n = self.gru(self.embed_norm(self.embed(x)))
        feat = self.shared(h_n[-1])
        R_mat = rot6d_to_rotmat_torch(self.rot_head(self.rot_blocks(feat)))
        t = self.trans_head(self.trans_blocks(feat))
        return torch.cat([R_mat.reshape(-1, 9), t], dim=1)


class IMUCameraPoseMLP6D(nn.Module):
    """Ablation baseline: Flattened MLP regressor with 6D rotation representation."""
    def __init__(self, seq_len: int = 60, input_dim: int = 21, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        in_dim = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rot_head = nn.Linear(hidden_dim, 6)
        self.trans_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        feat = self.net(x.view(B, -1))
        r6 = self.rot_head(feat)
        trans = self.trans_head(feat)
        R_mat = rot6d_to_rotmat_torch(r6)
        return R_mat, trans, r6
