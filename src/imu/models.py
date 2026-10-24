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
    """Residual block with LayerNorm and GELU activation."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class IMUCameraPoseGRU6D(nn.Module):
    """
    Proposed IMU-to-Camera relative-pose regressor (Fig. 3 in manuscript).
    
    Architecture:
      - Bidirectional or Unidirectional GRU temporal encoder
      - Shared bottleneck with LayerNorm
      - Decoupled rotation head (predicting 6D continuous representation -> SO(3))
      - Decoupled translation head (predicting 3D displacement vector)
    """
    def __init__(
        self,
        input_dim: int = 21,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        enc_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.bottleneck = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Decoupled Rotation Head: outputs 6D continuous representation
        self.rot_head = nn.Sequential(
            _ResidualMLPBlockLN(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, 6),
        )

        # Decoupled Translation Head: outputs 3D translation
        self.trans_head = nn.Sequential(
            _ResidualMLPBlockLN(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: IMU sequence tensor of shape (B, T, input_dim)

        Returns:
            R_mat: 3x3 SO(3) rotation matrices (B, 3, 3)
            trans: 3D translation vectors (B, 3)
            r6: Raw 6D rotation codes (B, 6)
        """
        B, T, D = x.shape
        proj = self.input_proj(x)
        out, h_n = self.gru(proj)

        if self.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            h_last = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h_last = h_n[-1]

        z = self.bottleneck(h_last)
        r6 = self.rot_head(z)
        trans = self.trans_head(z)
        R_mat = rot6d_to_rotmat_torch(r6)

        return R_mat, trans, r6


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
