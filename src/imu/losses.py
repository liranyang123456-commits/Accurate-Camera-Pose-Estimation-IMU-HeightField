"""
Geodesic and Adaptive Uncertainty Losses for Pose Regression.
Formulates Kendall multi-task uncertainty weighting between SO(3) rotation
and translation errors (Equation 9 in manuscript).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PoseGeodesicLoss(nn.Module):
    """
    Direct combination of SO(3) geodesic loss and Huber translation loss.
    """
    def __init__(self, w_rot: float = 1.0, w_trans: float = 1.0, huber_delta: float = 1.0):
        super().__init__()
        self.w_rot = float(w_rot)
        self.w_trans = float(w_trans)
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(
        self,
        R_pred: torch.Tensor,
        t_pred: torch.Tensor,
        R_gt: torch.Tensor,
        t_gt: torch.Tensor
    ) -> torch.Tensor:
        # Geodesic rotation distance
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_theta = torch.clamp((tr - 1.0) / 2.0, -0.999999, 0.999999)
        loss_rot = torch.acos(cos_theta).mean()

        # Translation loss
        loss_trans = self.huber(t_pred, t_gt)

        return self.w_rot * loss_rot + self.w_trans * loss_trans


class AdaptivePoseGeodesicLoss(nn.Module):
    """
    Multi-task adaptive loss with learnable homoscedastic uncertainty parameters:
      L = exp(-s_r) * L_rot + s_r + exp(-s_t) * L_trans + s_t
    """
    def __init__(self, init_s_r: float = 0.0, init_s_t: float = 0.0, huber_delta: float = 1.0):
        super().__init__()
        self.s_r = nn.Parameter(torch.tensor(float(init_s_r), dtype=torch.float32))
        self.s_t = nn.Parameter(torch.tensor(float(init_s_t), dtype=torch.float32))
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(
        self,
        R_pred: torch.Tensor,
        t_pred: torch.Tensor,
        R_gt: torch.Tensor,
        t_gt: torch.Tensor
    ) -> torch.Tensor:
        # Geodesic angle in radians
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_theta = torch.clamp((tr - 1.0) / 2.0, -0.999999, 0.999999)
        loss_rot = torch.acos(cos_theta).mean()

        # Translation loss
        loss_trans = self.huber(t_pred, t_gt)

        # Uncertainty weighted combination
        prec_r = torch.exp(-self.s_r)
        prec_t = torch.exp(-self.s_t)

        return prec_r * loss_rot + self.s_r + prec_t * loss_trans + self.s_t
