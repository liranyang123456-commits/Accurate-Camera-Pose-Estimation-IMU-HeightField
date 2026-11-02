#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training module of the evaluated GRU6D run: pose scaler, losses, regressors, and trainer. train_inertial_baselines.py
imports the regressors from here, and the evaluated checkpoint unpickles against these classes.

IMU到相机位姿MLP网络拟合
通过MLP网络学习IMU数据与相机RT矩阵之间的映射关系
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import json
import os
import glob
from datetime import datetime
import cv2
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class PoseScalerRT:
    """
    仅对平移 t 做标准化，旋转矩阵 R(flat 9维) 保持原值不变。

    用途：
      - 旋转元素本身在 [-1,1] 附近，通常无需 StandardScaler；
      - 平移尺度可能与旋转差异很大，单独标准化更利于训练且不破坏旋转矩阵语义。
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.t_mean_: Optional[np.ndarray] = None
        self.t_scale_: Optional[np.ndarray] = None

    def fit(self, poses: np.ndarray):
        arr = np.asarray(poses, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or int(arr.shape[1]) < 12:
            raise ValueError(f"poses 形状应为 (N,12)，但得到 {arr.shape}")
        t = arr[:, 9:12]
        t_mean = t.mean(axis=0)
        t_std = t.std(axis=0)
        t_std = np.where(t_std < self.eps, 1.0, t_std)
        self.t_mean_ = t_mean.astype(np.float64)
        self.t_scale_ = t_std.astype(np.float64)
        return self

    def transform(self, poses: np.ndarray) -> np.ndarray:
        if self.t_mean_ is None or self.t_scale_ is None:
            raise RuntimeError("PoseScalerRT 尚未 fit")
        arr = np.asarray(poses, dtype=np.float64)
        single = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            single = True
        if arr.ndim != 2 or int(arr.shape[1]) < 12:
            raise ValueError(f"poses 形状应为 (N,12)，但得到 {arr.shape}")
        out = arr.copy()
        out[:, 9:12] = (out[:, 9:12] - self.t_mean_) / self.t_scale_
        return out[0] if single else out

    def inverse_transform(self, poses: np.ndarray) -> np.ndarray:
        if self.t_mean_ is None or self.t_scale_ is None:
            raise RuntimeError("PoseScalerRT 尚未 fit")
        arr = np.asarray(poses, dtype=np.float64)
        single = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            single = True
        if arr.ndim != 2 or int(arr.shape[1]) < 12:
            raise ValueError(f"poses 形状应为 (N,12)，但得到 {arr.shape}")
        out = arr.copy()
        out[:, 9:12] = out[:, 9:12] * self.t_scale_ + self.t_mean_
        return out[0] if single else out

    def fit_transform(self, poses: np.ndarray) -> np.ndarray:
        return self.fit(poses).transform(poses)


def _project_to_so3_torch(Rm: torch.Tensor) -> torch.Tensor:
    """SVD 投影到 SO(3)（支持 batch）。Rm: (B,3,3)"""
    U, _, Vh = torch.linalg.svd(Rm)
    R = U @ Vh
    det = torch.det(R)
    det_sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    U2 = U.clone()
    U2[:, :, -1] = U2[:, :, -1] * det_sign.unsqueeze(-1)
    return U2 @ Vh


def rot6d_to_rotmat_torch(x6: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    6D 旋转表示 -> 3x3 旋转矩阵（Zhou et al., 2019）
    x6: (B,6) -> R: (B,3,3)
    """
    if x6.ndim != 2 or int(x6.shape[1]) != 6:
        raise ValueError(f"x6 形状应为 (B,6)，但得到 {tuple(x6.shape)}")
    a1 = x6[:, 0:3]
    a2 = x6[:, 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=1, eps=float(eps))
    dot = (b1 * a2).sum(dim=1, keepdim=True)
    a2_ortho = a2 - dot * b1
    b2 = torch.nn.functional.normalize(a2_ortho, dim=1, eps=float(eps))
    b3 = torch.cross(b1, b2, dim=1)
    R = torch.stack([b1, b2, b3], dim=2)  # columns
    return R


class AdaptivePoseGeodesicLoss(nn.Module):
    """
    自适应损失权重（Kendall 多任务不确定性权重思想）
      total = exp(-s_r)*L_rot + s_r + exp(-s_t)*L_trans + s_t + ortho/det penalties

    说明：
      - L_rot 用 SO(3) 的测地角（弧度），更对齐“旋转角度误差”指标
      - s_r/s_t 是可学习参数，会自动平衡旋转与平移损失
    """

    def __init__(
        self,
        *,
        trans_loss: str = "smoothl1",
        rot_scale: float = 1.0,
        trans_scale: float = 1.0,
        ortho_weight: float = 0.0,
        det_weight: float = 0.0,
        project_to_so3: bool = True,
        ste_projection: bool = True,
        acos_eps: float = 1e-6,
        init_log_var_rot: float = 0.0,
        init_log_var_trans: float = 0.0,
    ):
        super().__init__()
        self.rot_scale = float(rot_scale)
        self.trans_scale = float(trans_scale)
        self.ortho_weight = float(ortho_weight)
        self.det_weight = float(det_weight)
        self.project_to_so3 = bool(project_to_so3)
        self.ste_projection = bool(ste_projection)
        self.acos_eps = float(max(1e-12, acos_eps))

        tl = str(trans_loss).strip().lower()
        if tl not in ("smoothl1", "mse"):
            tl = "smoothl1"
        self._trans_crit = nn.SmoothL1Loss() if tl == "smoothl1" else nn.MSELoss()

        self.log_var_rot = nn.Parameter(torch.tensor(float(init_log_var_rot)))
        self.log_var_trans = nn.Parameter(torch.tensor(float(init_log_var_trans)))

    def forward(self, pred_pose: torch.Tensor, true_pose: torch.Tensor) -> torch.Tensor:
        if pred_pose.ndim != 2 or int(pred_pose.shape[1]) < 12:
            raise ValueError(f"pred_pose 形状应为 (B,12)，但得到 {tuple(pred_pose.shape)}")
        if true_pose.ndim != 2 or int(true_pose.shape[1]) < 12:
            raise ValueError(f"true_pose 形状应为 (B,12)，但得到 {tuple(true_pose.shape)}")

        pred_R_raw = pred_pose[:, :9].reshape(-1, 3, 3)
        true_R_raw = true_pose[:, :9].reshape(-1, 3, 3)
        if self.project_to_so3:
            # 注意：SVD 投影在旋转矩阵处奇异值重复，直接反传可能产生 NaN 梯度；
            # 这里默认用“直通估计(ste)”：前向使用投影值，反向视作恒等映射。
            pred_R_proj = _project_to_so3_torch(pred_R_raw)
            if bool(self.ste_projection):
                pred_R = pred_R_raw + (pred_R_proj - pred_R_raw).detach()
            else:
                pred_R = pred_R_proj
            true_R = true_R_raw
        else:
            pred_R = pred_R_raw
            true_R = true_R_raw

        R_diff = pred_R.transpose(1, 2) @ true_R
        tr = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + self.acos_eps, 1.0 - self.acos_eps)
        rot_loss = torch.acos(cos_theta).mean() * float(self.rot_scale)

        pred_t = pred_pose[:, 9:12]
        true_t = true_pose[:, 9:12]
        trans_loss = self._trans_crit(pred_t, true_t) * float(self.trans_scale)

        # uncertainty weighting
        loss = torch.exp(-self.log_var_rot) * rot_loss + self.log_var_rot
        loss = loss + torch.exp(-self.log_var_trans) * trans_loss + self.log_var_trans

        if self.ortho_weight > 0.0:
            I = torch.eye(3, device=pred_R_raw.device, dtype=pred_R_raw.dtype).unsqueeze(0)
            RtR = pred_R_raw.transpose(1, 2) @ pred_R_raw
            ortho = (RtR - I).pow(2).mean()
            loss = loss + float(self.ortho_weight) * ortho

        if self.det_weight > 0.0:
            det = torch.det(pred_R_raw)
            det_pen = (det - 1.0).pow(2).mean()
            loss = loss + float(self.det_weight) * det_pen

        return loss

    def get_current_weights(self) -> dict:
        """返回当前等效权重（exp(-log_var)）。"""
        with torch.no_grad():
            w_rot = float(torch.exp(-self.log_var_rot).detach().cpu().item())
            w_trans = float(torch.exp(-self.log_var_trans).detach().cpu().item())
            return {"w_rot": w_rot, "w_trans": w_trans, "log_var_rot": float(self.log_var_rot.detach().cpu().item()), "log_var_trans": float(self.log_var_trans.detach().cpu().item())}


class PoseGeodesicLoss(nn.Module):
    """
    面向旋转角度误差的损失：
      - rotation: SO(3) geodesic angle loss（弧度）
      - translation: SmoothL1 或 MSE（建议配合 PoseScalerRT 标准化后的 t）
      - 可选：对 raw 预测旋转加正交/行列式惩罚，逼近合法旋转矩阵
    """

    def __init__(
        self,
        *,
        rot_weight: float = 1.0,
        trans_weight: float = 1.0,
        trans_loss: str = "smoothl1",
        ortho_weight: float = 0.0,
        det_weight: float = 0.0,
        project_to_so3: bool = True,
        ste_projection: bool = True,
        acos_eps: float = 1e-6,
    ):
        super().__init__()
        self.rot_weight = float(rot_weight)
        self.trans_weight = float(trans_weight)
        self.ortho_weight = float(ortho_weight)
        self.det_weight = float(det_weight)
        self.project_to_so3 = bool(project_to_so3)
        self.ste_projection = bool(ste_projection)
        self.acos_eps = float(max(1e-12, acos_eps))

        tl = str(trans_loss).strip().lower()
        if tl not in ("smoothl1", "mse"):
            tl = "smoothl1"
        self._trans_loss_name = tl
        self._trans_crit = nn.SmoothL1Loss() if tl == "smoothl1" else nn.MSELoss()

    def forward(self, pred_pose: torch.Tensor, true_pose: torch.Tensor) -> torch.Tensor:
        if pred_pose.ndim != 2 or int(pred_pose.shape[1]) < 12:
            raise ValueError(f"pred_pose 形状应为 (B,12)，但得到 {tuple(pred_pose.shape)}")
        if true_pose.ndim != 2 or int(true_pose.shape[1]) < 12:
            raise ValueError(f"true_pose 形状应为 (B,12)，但得到 {tuple(true_pose.shape)}")

        pred_R_raw = pred_pose[:, :9].reshape(-1, 3, 3)
        true_R_raw = true_pose[:, :9].reshape(-1, 3, 3)

        if self.project_to_so3:
            pred_R_proj = _project_to_so3_torch(pred_R_raw)
            if bool(self.ste_projection):
                pred_R = pred_R_raw + (pred_R_proj - pred_R_raw).detach()
            else:
                pred_R = pred_R_proj
            true_R = true_R_raw
        else:
            pred_R = pred_R_raw
            true_R = true_R_raw

        R_diff = pred_R.transpose(1, 2) @ true_R
        tr = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + self.acos_eps, 1.0 - self.acos_eps)
        rot_loss = torch.acos(cos_theta).mean()

        pred_t = pred_pose[:, 9:12]
        true_t = true_pose[:, 9:12]
        trans_loss = self._trans_crit(pred_t, true_t)

        loss = self.rot_weight * rot_loss + self.trans_weight * trans_loss

        if self.ortho_weight > 0.0:
            I = torch.eye(3, device=pred_R_raw.device, dtype=pred_R_raw.dtype).unsqueeze(0)
            RtR = pred_R_raw.transpose(1, 2) @ pred_R_raw
            ortho = (RtR - I).pow(2).mean()
            loss = loss + self.ortho_weight * ortho

        if self.det_weight > 0.0:
            det = torch.det(pred_R_raw)
            det_pen = (det - 1.0).pow(2).mean()
            loss = loss + self.det_weight * det_pen

        return loss


class IMUCameraDataset(Dataset):
    """IMU到相机位姿数据集"""
    
    def __init__(self, imu_data, camera_poses, scaler_imu=None, scaler_pose=None):
        """
        Args:
            imu_data: numpy array, shape (N, 17) - IMU特征
            camera_poses: numpy array, shape (N, 12) - 相机位姿(R矩阵9维 + t向量3维)
            scaler_imu: 用于IMU数据标准化的scaler
            scaler_pose: 用于位姿数据标准化的scaler
        """
        self.imu_data = torch.FloatTensor(imu_data)
        self.camera_poses = torch.FloatTensor(camera_poses)
        
        # 数据标准化
        if scaler_imu is not None:
            self.imu_data = torch.FloatTensor(scaler_imu.transform(imu_data))
        if scaler_pose is not None:
            self.camera_poses = torch.FloatTensor(scaler_pose.transform(camera_poses))
    
    def __len__(self):
        return len(self.imu_data)
    
    def __getitem__(self, idx):
        return self.imu_data[idx], self.camera_poses[idx]

class IMUCameraMLP(nn.Module):
    """IMU到相机位姿的MLP网络"""
    
    def __init__(self, input_dim=17, hidden_dims=[256, 256, 128], output_dim=12, dropout=0.2):
        super(IMUCameraMLP, self).__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

    def get_config(self) -> dict:
        """用于保存/恢复模型结构参数。"""
        return {
            "class": self.__class__.__name__,
            "input_dim": int(self.input_dim),
            "hidden_dims": list(self.hidden_dims),
            "output_dim": int(self.output_dim),
            "dropout": float(self.dropout),
        }


class IMUCameraPoseMLP6D(nn.Module):
    """
    旋转表示改造版：
      - 网络回归 rot6(6D) + t(3D)
      - 前向中用 rot6 -> R(3x3)（可微正交化），最终输出仍为 [R_flat(9), t(3)] 共 12 维

    好处：
      - 预测旋转天然更接近合法 SO(3)，更利于降低旋转角度误差
      - 下游（保存/可视化/评估）保持与旧版 12 维格式兼容
    """

    def __init__(self, input_dim=17, hidden_dims=[256, 256, 128], dropout=0.2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.output_dim = 12
        self.dropout = float(dropout)

        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, int(hidden_dim)),
                nn.BatchNorm1d(int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            ])
            prev_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)

        self.rot_head = nn.Linear(prev_dim, 6)
        self.trans_head = nn.Linear(prev_dim, 3)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        r6 = self.rot_head(feat)
        t = self.trans_head(feat)
        Rm = rot6d_to_rotmat_torch(r6)  # (B,3,3)
        R_flat = Rm.reshape(-1, 9)
        return torch.cat([R_flat, t], dim=1)

    def get_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "input_dim": int(self.input_dim),
            "hidden_dims": list(self.hidden_dims),
            "output_dim": int(self.output_dim),
            "dropout": float(self.dropout),
        }


class _ResidualMLPBlockLN(nn.Module):
    """LayerNorm + GELU 的残差 MLP Block（更适合小 batch）。"""

    def __init__(self, dim: int, *, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        d = int(dim)
        h = int(max(1, d * int(expansion)))
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(h, d)
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


def _project_to_so3_gs_torch(Rm: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Gram-Schmidt 投影到 SO(3)（支持 batch），避免在训练/评估中依赖 SVD。Rm: (B,3,3)"""
    if Rm.ndim != 3 or int(Rm.shape[-2]) != 3 or int(Rm.shape[-1]) != 3:
        raise ValueError(f"Rm 形状应为 (B,3,3)，但得到 {tuple(Rm.shape)}")
    a1 = Rm[:, :, 0]
    a2 = Rm[:, :, 1]
    b1 = torch.nn.functional.normalize(a1, dim=1, eps=float(eps))
    dot = (b1 * a2).sum(dim=1, keepdim=True)
    a2o = a2 - dot * b1
    b2 = torch.nn.functional.normalize(a2o, dim=1, eps=float(eps))
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


class IMUCameraPoseGRU6D(nn.Module):
    """
    时间序列版（窗口输入）：
      - 输入: (B,T,17)
      - Temporal Encoder: GRU
      - 输出: rot6 + t -> [R_flat(9), t(3)]

    结构特点：
      - LayerNorm（替代 BatchNorm，更稳）
      - Rot/Trans 分支头（更独立）
      - Residual MLP Blocks（更易训练）
    """

    expects_sequence: bool = True

    def __init__(
        self,
        *,
        input_dim: int = 17,
        embed_dim: int = 128,
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 2,
        head_dim: int = 256,
        head_blocks: int = 2,
        dropout: float = 0.2,
        aux_imu_end_dt: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru_num_layers = int(gru_num_layers)
        self.head_dim = int(head_dim)
        self.head_blocks = int(head_blocks)
        self.dropout = float(dropout)
        self.aux_imu_end_dt = bool(aux_imu_end_dt)
        self.output_dim = 12

        self.embed = nn.Linear(self.input_dim, self.embed_dim)
        self.embed_norm = nn.LayerNorm(self.embed_dim)

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=self.gru_num_layers,
            batch_first=True,
            dropout=float(self.dropout) if int(self.gru_num_layers) > 1 else 0.0,
        )

        # shared -> smaller
        self.shared = nn.Sequential(
            nn.LayerNorm(self.gru_hidden_dim),
            nn.Linear(self.gru_hidden_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(float(self.dropout)),
        )

        self.rot_blocks = nn.Sequential(
            *[_ResidualMLPBlockLN(self.head_dim, expansion=4, dropout=float(self.dropout)) for _ in range(int(max(0, self.head_blocks)))]
        )
        self.trans_blocks = nn.Sequential(
            *[_ResidualMLPBlockLN(self.head_dim, expansion=4, dropout=float(self.dropout)) for _ in range(int(max(0, self.head_blocks)))]
        )

        self.rot_head = nn.Linear(self.head_dim, 6)
        self.trans_head = nn.Linear(self.head_dim, 3)
        self.aux_head = nn.Linear(self.head_dim, 1) if bool(self.aux_imu_end_dt) else None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or int(x.shape[-1]) != int(self.input_dim):
            raise ValueError(f"输入应为 (B,T,{self.input_dim})，但得到 {tuple(x.shape)}")
        h = self.embed(x)
        h = self.embed_norm(h)
        out, hn = self.gru(h)  # hn: (L,B,H)
        feat = hn[-1]  # (B,H)
        feat = self.shared(feat)
        rot_feat = self.rot_blocks(feat)
        trans_feat = self.trans_blocks(feat)
        r6 = self.rot_head(rot_feat)
        t = self.trans_head(trans_feat)
        Rm = rot6d_to_rotmat_torch(r6)
        pose12 = torch.cat([Rm.reshape(-1, 9), t], dim=1)
        if bool(self.aux_imu_end_dt) and (self.aux_head is not None):
            aux = self.aux_head(feat)  # (B,1)
            return pose12, aux
        return pose12

    def get_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "input_dim": int(self.input_dim),
            "embed_dim": int(self.embed_dim),
            "gru_hidden_dim": int(self.gru_hidden_dim),
            "gru_num_layers": int(self.gru_num_layers),
            "head_dim": int(self.head_dim),
            "head_blocks": int(self.head_blocks),
            "dropout": float(self.dropout),
            "output_dim": int(self.output_dim),
            "aux_imu_end_dt": bool(self.aux_imu_end_dt),
        }


class _TCNResBlock(nn.Module):
    def __init__(self, ch: int, *, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        k = int(max(1, kernel_size))
        d = int(max(1, dilation))
        pad = (k - 1) * d // 2
        self.conv1 = nn.Conv1d(int(ch), int(ch), kernel_size=k, dilation=d, padding=int(pad))
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(int(ch), int(ch), kernel_size=k, dilation=d, padding=int(pad))
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(float(dropout))
        self.norm = nn.BatchNorm1d(int(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.act1(h)
        h = self.drop1(h)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.drop2(h)
        h = self.norm(h)
        return x + h


class IMUCameraPoseTCN6D(nn.Module):
    expects_sequence: bool = True

    def __init__(
        self,
        *,
        input_dim: int = 21,
        embed_dim: int = 128,
        channels: int = 256,
        num_blocks: int = 4,
        kernel_size: int = 3,
        head_dim: int = 256,
        head_blocks: int = 2,
        dropout: float = 0.2,
        aux_imu_end_dt: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        self.head_dim = int(head_dim)
        self.head_blocks = int(head_blocks)
        self.dropout = float(dropout)
        self.aux_imu_end_dt = bool(aux_imu_end_dt)
        self.output_dim = 12

        self.embed = nn.Linear(self.input_dim, self.embed_dim)
        self.embed_norm = nn.LayerNorm(self.embed_dim)
        self.proj = nn.Conv1d(self.embed_dim, self.channels, kernel_size=1)

        blocks = []
        for i in range(int(max(1, self.num_blocks))):
            dil = 2 ** int(i)
            blocks.append(_TCNResBlock(self.channels, kernel_size=self.kernel_size, dilation=dil, dropout=float(self.dropout)))
        self.tcn = nn.Sequential(*blocks)

        self.shared = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.head_dim),
            nn.GELU(),
            nn.Dropout(float(self.dropout)),
        )
        self.rot_blocks = nn.Sequential(
            *[_ResidualMLPBlockLN(self.head_dim, expansion=4, dropout=float(self.dropout)) for _ in range(int(max(0, self.head_blocks)))]
        )
        self.trans_blocks = nn.Sequential(
            *[_ResidualMLPBlockLN(self.head_dim, expansion=4, dropout=float(self.dropout)) for _ in range(int(max(0, self.head_blocks)))]
        )
        self.rot_head = nn.Linear(self.head_dim, 6)
        self.trans_head = nn.Linear(self.head_dim, 3)
        self.aux_head = nn.Linear(self.head_dim, 1) if bool(self.aux_imu_end_dt) else None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or int(x.shape[-1]) != int(self.input_dim):
            raise ValueError(f"输入应为 (B,T,{self.input_dim})，但得到 {tuple(x.shape)}")
        h = self.embed(x)  # (B,T,E)
        h = self.embed_norm(h)
        h = h.transpose(1, 2)  # (B,E,T)
        h = self.proj(h)  # (B,C,T)
        h = self.tcn(h)  # (B,C,T)
        feat0 = h.mean(dim=2)  # (B,C)
        feat = self.shared(feat0)  # (B,head_dim)
        rot_feat = self.rot_blocks(feat)
        trans_feat = self.trans_blocks(feat)
        r6 = self.rot_head(rot_feat)
        t = self.trans_head(trans_feat)
        Rm = rot6d_to_rotmat_torch(r6)
        pose12 = torch.cat([Rm.reshape(-1, 9), t], dim=1)
        if bool(self.aux_imu_end_dt) and (self.aux_head is not None):
            aux = self.aux_head(feat)
            return pose12, aux
        return pose12

    def get_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "input_dim": int(self.input_dim),
            "embed_dim": int(self.embed_dim),
            "channels": int(self.channels),
            "num_blocks": int(self.num_blocks),
            "kernel_size": int(self.kernel_size),
            "head_dim": int(self.head_dim),
            "head_blocks": int(self.head_blocks),
            "dropout": float(self.dropout),
            "output_dim": int(self.output_dim),
            "aux_imu_end_dt": bool(self.aux_imu_end_dt),
        }


def _sinusoidal_pos_encoding(T: int, d: int, *, device: torch.device) -> torch.Tensor:
    T = int(max(1, T))
    d = int(max(1, d))
    pe = torch.zeros((T, d), device=device, dtype=torch.float32)
    pos = torch.arange(0, T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / float(d)))
    pe[:, 0::2] = torch.sin(pos * div)
    if int(d) > 1:
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class IMUCameraPoseTransformer6D(nn.Module):
    expects_sequence: bool = True

    def __init__(
        self,
        *,
        input_dim: int = 21,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        head_dim: int = 256,
        head_blocks: int = 2,
        dropout: float = 0.2,
        aux_imu_end_dt: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.head_dim = int(head_dim)
        self.head_blocks = int(head_blocks)
        self.dropout = float(dropout)
        self.aux_imu_end_dt = bool(aux_imu_end_dt)
        self.output_dim = 12

        self.in_proj = nn.Linear(self.input_dim, self.model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(self.model_dim),
            nhead=int(max(1, self.num_heads)),
            dim_feedforward=int(max(int(self.model_dim) * 2, self.ffn_dim)),
            dropout=float(self.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(max(1, self.num_layers)))
        self.out_norm = nn.LayerNorm(self.model_dim)

        self.shared = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(float(self.dropout)),
        )
        self.rot_blocks = nn.Sequential(
            *[_ResidualMLPBlockLN(self.head_dim, expansion=4, dropout=float(self.dropout)) for _ in range(int(max(0, self.head_blocks)))]
        )
        self.trans_blocks = nn.Sequential(
            *[_ResidualMLPBlockLN(self.head_dim, expansion=4, dropout=float(self.dropout)) for _ in range(int(max(0, self.head_blocks)))]
        )
        self.rot_head = nn.Linear(self.head_dim, 6)
        self.trans_head = nn.Linear(self.head_dim, 3)
        self.aux_head = nn.Linear(self.head_dim, 1) if bool(self.aux_imu_end_dt) else None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or int(x.shape[-1]) != int(self.input_dim):
            raise ValueError(f"输入应为 (B,T,{self.input_dim})，但得到 {tuple(x.shape)}")
        T = int(x.shape[1])
        h = self.in_proj(x)  # (B,T,C)
        pe = _sinusoidal_pos_encoding(int(T), int(self.model_dim), device=h.device).unsqueeze(0)  # (1,T,C)
        h = h + pe
        h = self.encoder(h)
        h = self.out_norm(h)
        feat0 = h.mean(dim=1)  # (B,C)
        feat = self.shared(feat0)  # (B,head_dim)
        rot_feat = self.rot_blocks(feat)
        trans_feat = self.trans_blocks(feat)
        r6 = self.rot_head(rot_feat)
        t = self.trans_head(trans_feat)
        Rm = rot6d_to_rotmat_torch(r6)
        pose12 = torch.cat([Rm.reshape(-1, 9), t], dim=1)
        if bool(self.aux_imu_end_dt) and (self.aux_head is not None):
            aux = self.aux_head(feat)
            return pose12, aux
        return pose12

    def get_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "input_dim": int(self.input_dim),
            "model_dim": int(self.model_dim),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "ffn_dim": int(self.ffn_dim),
            "head_dim": int(self.head_dim),
            "head_blocks": int(self.head_blocks),
            "dropout": float(self.dropout),
            "output_dim": int(self.output_dim),
            "aux_imu_end_dt": bool(self.aux_imu_end_dt),
        }


class IMUCameraMLPTrainer:
    """IMU到相机位姿MLP训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler_imu = StandardScaler()
        self.scaler_pose = StandardScaler()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(self, imu_data, camera_poses, test_size=0.2, random_state=42):
        """准备训练数据"""
        print(f"准备数据: {len(imu_data)} 个样本")

        # 先分割再拟合 scaler，避免数据泄漏
        idx = np.arange(len(imu_data))
        idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)

        imu_train = np.asarray(imu_data)[idx_train]
        imu_test = np.asarray(imu_data)[idx_test]
        pose_train = np.asarray(camera_poses)[idx_train]
        pose_test = np.asarray(camera_poses)[idx_test]

        self.scaler_imu.fit(imu_train)
        self.scaler_pose.fit(pose_train)

        X_train = self.scaler_imu.transform(imu_train)
        X_test = self.scaler_imu.transform(imu_test)
        y_train = self.scaler_pose.transform(pose_train)
        y_test = self.scaler_pose.transform(pose_test)
        
        # 创建数据集
        train_dataset = IMUCameraDataset(X_train, y_train)
        test_dataset = IMUCameraDataset(X_test, y_test)
        
        return train_dataset, test_dataset
    
    def train_epoch(
        self,
        dataloader,
        optimizer,
        criterion,
        *,
        grad_clip_norm: Optional[float] = 1.0,
        aux_weight: float = 0.0,
    ):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                imu_data = batch[0]
                poses = batch[1]
                aux_t = batch[2] if len(batch) >= 3 else None
            else:
                raise ValueError("Dataloader must yield (imu, pose) or (imu, pose, aux).")

            imu_data = imu_data.to(self.device)
            poses = poses.to(self.device)
            if aux_t is not None:
                aux_t = aux_t.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(imu_data)
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                pose_out = outputs[0]
                aux_out = outputs[1]
            else:
                pose_out = outputs
                aux_out = None

            loss = criterion(pose_out, poses)
            if aux_t is not None and aux_out is not None and float(aux_weight) > 0.0:
                aux_target = torch.abs(aux_t.reshape(-1, 1).to(aux_out.dtype))
                aux_pred = aux_out.reshape(-1, 1)
                aux_loss = torch.nn.functional.smooth_l1_loss(aux_pred, aux_target)
                loss = loss + float(aux_weight) * aux_loss
            if not torch.isfinite(loss):
                # 尽量在参数被污染前中断，方便定位数值问题
                if isinstance(pose_out, torch.Tensor):
                    out_ok = bool(torch.isfinite(pose_out).all().item())
                else:
                    out_ok = True
                pose_ok = bool(torch.isfinite(poses).all().item())
                raise FloatingPointError(f"Non-finite loss detected. outputs_finite={out_ok} poses_finite={pose_ok}")
            loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip_norm))
                if hasattr(gn, "detach"):
                    gn_val = gn.detach()
                else:
                    gn_val = gn
                if not torch.isfinite(gn_val).all():
                    raise FloatingPointError("Non-finite gradient norm detected (possible NaN/Inf gradients).")
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    imu_data = batch[0]
                    poses = batch[1]
                else:
                    raise ValueError("Dataloader must yield (imu, pose) or (imu, pose, aux).")

                imu_data = imu_data.to(self.device)
                poses = poses.to(self.device)
                
                outputs = self.model(imu_data)
                if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                    pose_out = outputs[0]
                else:
                    pose_out = outputs
                loss = criterion(pose_out, poses)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(
        self,
        train_dataset,
        val_dataset,
        epochs=1000,
        batch_size=32,
        learning_rate=0.001,
        patience=50,  # <=0 或 None 表示不早停
        weight_decay: float = 1e-5,
        grad_clip_norm: Optional[float] = 1.0,
        num_workers: int = 0,
        criterion=None,
        best_model_path: str = "best_model.pth",
        log_metrics_every: int = 0,
        metrics_project_to_so3: bool = True,
        use_lr_scheduler: bool = True,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
        log_current_lr: bool = True,
        aux_weight: float = 0.0,
    ):
        """训练模型"""
        print(f"开始训练，设备: {self.device}")
        
        # 创建数据加载器
        pin_memory = str(self.device).startswith("cuda")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(max(0, num_workers)),
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(max(0, num_workers)),
            pin_memory=pin_memory,
        )
        
        # 优化器和损失函数
        if criterion is None:
            criterion = nn.MSELoss()
        if isinstance(criterion, nn.Module):
            criterion = criterion.to(self.device)

        opt_params = list(self.model.parameters())
        if isinstance(criterion, nn.Module):
            crit_params = [p for p in criterion.parameters() if p.requires_grad]
            if crit_params:
                opt_params.extend(crit_params)

        optimizer = optim.AdamW(opt_params, lr=float(learning_rate), weight_decay=float(weight_decay))

        # 可配置的 LR scheduler（plateau）
        scheduler = None
        if bool(use_lr_scheduler):
            sp = int(max(0, scheduler_patience))
            fac = float(scheduler_factor)
            if not (0.0 < fac < 1.0):
                fac = 0.5
            mlr = float(max(0.0, scheduler_min_lr))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=int(sp),
                factor=float(fac),
                min_lr=float(mlr),
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        use_early_stop = (patience is not None) and (int(patience) > 0)
        patience_i = int(patience) if patience is not None else 0

        # 训练日志：打印 val 角度/平移误差（不只看 loss）
        log_every = int(max(0, log_metrics_every))
        use_metrics = bool(log_every > 0)
        rad2deg = 180.0 / float(np.pi)

        # 预先构造 inverse_transform 所需参数（支持 PoseScalerRT / StandardScaler）
        sp = self.scaler_pose
        use_pose_rt = hasattr(sp, "t_mean_") and hasattr(sp, "t_scale_") and (getattr(sp, "t_mean_", None) is not None)
        use_std_scaler = hasattr(sp, "mean_") and hasattr(sp, "scale_") and (getattr(sp, "mean_", None) is not None)
        t_mean = None
        t_scale = None
        mean12 = None
        scale12 = None

        if use_pose_rt:
            t_mean = torch.tensor(np.asarray(sp.t_mean_, dtype=np.float32), device=self.device)
            t_scale = torch.tensor(np.asarray(sp.t_scale_, dtype=np.float32), device=self.device)
        elif use_std_scaler:
            mean12 = torch.tensor(np.asarray(sp.mean_, dtype=np.float32), device=self.device)
            scale12 = torch.tensor(np.asarray(sp.scale_, dtype=np.float32), device=self.device)

        def _inv_pose(p: torch.Tensor) -> torch.Tensor:
            if not (p.ndim == 2 and int(p.shape[1]) >= 12):
                return p
            if use_pose_rt and t_mean is not None and t_scale is not None:
                out = p.clone()
                out[:, 9:12] = out[:, 9:12] * t_scale + t_mean
                return out
            if use_std_scaler and mean12 is not None and scale12 is not None and int(p.shape[1]) == int(mean12.numel()):
                return p * scale12 + mean12
            return p

        def _val_metrics(loader) -> Optional[tuple[float, float, float, float]]:
            rot_all = []
            trans_all = []
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        imu_b = batch[0]
                        pose_b = batch[1]
                    else:
                        raise ValueError("Dataloader must yield (imu, pose) or (imu, pose, aux).")
                    imu_b = imu_b.to(self.device)
                    pose_b = pose_b.to(self.device)
                    pred_b = self.model(imu_b)
                    if isinstance(pred_b, (list, tuple)) and len(pred_b) >= 2:
                        pred_b = pred_b[0]

                    pred_u = _inv_pose(pred_b)
                    true_u = _inv_pose(pose_b)

                    pred_R = pred_u[:, :9].reshape(-1, 3, 3)
                    true_R = true_u[:, :9].reshape(-1, 3, 3)
                    if bool(metrics_project_to_so3):
                        pred_R = _project_to_so3_gs_torch(pred_R)
                        true_R = _project_to_so3_gs_torch(true_R)

                    trc = (pred_R * true_R).sum(dim=(1, 2))  # trace(Rp^T Rt)
                    cos_theta = (trc - 1.0) / 2.0
                    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
                    rot_deg = torch.acos(cos_theta) * float(rad2deg)

                    pred_t = pred_u[:, 9:12]
                    true_t = true_u[:, 9:12]
                    d = pred_t - true_t
                    trans = torch.sqrt(torch.sum(d * d, dim=1))

                    rot_all.append(rot_deg.detach().cpu())
                    trans_all.append(trans.detach().cpu())

            if not rot_all:
                return None
            rot_cat = torch.cat(rot_all, dim=0)
            trans_cat = torch.cat(trans_all, dim=0) if trans_all else torch.empty_like(rot_cat)
            return (
                float(rot_cat.mean().item()),
                float(rot_cat.std(unbiased=False).item()),
                float(trans_cat.mean().item()),
                float(trans_cat.std(unbiased=False).item()),
            )
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(
                train_loader,
                optimizer,
                criterion,
                grad_clip_norm=grad_clip_norm,
                aux_weight=float(aux_weight),
            )
            val_loss = self.validate(val_loader, criterion)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型（不刷屏）
                self.save_model(str(best_model_path), verbose=False)
            else:
                patience_counter += 1
            
            # 打印进度（可选打印物理指标：旋转角度/平移距离）
            print_now = (epoch % 50 == 0) or (epoch == epochs - 1) or (bool(use_metrics) and epoch % int(log_every) == 0)
            if bool(print_now):
                cur_lr = None
                try:
                    cur_lr = float(optimizer.param_groups[0].get("lr", float(learning_rate)))
                except Exception:
                    cur_lr = None
                msg = f"Epoch {epoch:4d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
                if bool(log_current_lr) and (cur_lr is not None):
                    msg += f" | lr={cur_lr:.2e}"
                if bool(use_metrics) and (epoch % int(log_every) == 0 or epoch == epochs - 1):
                    m = _val_metrics(val_loader)
                    if m is not None:
                        msg += f" | Val rot(deg)={m[0]:.2f}±{m[1]:.2f} trans={m[2]:.3f}±{m[3]:.3f}"
                print(msg)
            
            # 早停检查（patience<=0 表示不早停）
            if bool(use_early_stop) and patience_counter >= int(patience_i):
                print(f"早停于 epoch {epoch}")
                break
        
        print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
        return best_val_loss
    
    def predict(self, imu_data, *, project_to_so3: bool = False):
        """预测相机位姿（可选将旋转投影到 SO(3)）。"""
        self.model.eval()
        
        # 标准化输入数据
        imu_data_scaled = self.scaler_imu.transform(imu_data.reshape(1, -1))
        imu_tensor = torch.FloatTensor(imu_data_scaled).to(self.device)
        
        with torch.no_grad():
            output = self.model(imu_tensor)
            if isinstance(output, (list, tuple)) and len(output) >= 2:
                output = output[0]
            output_np = output.cpu().numpy()
        
        # 反标准化
        pose_pred = self.scaler_pose.inverse_transform(output_np)
        out = np.asarray(pose_pred[0], dtype=np.float64)
        if bool(project_to_so3) and out.size >= 12:
            Rm = out[:9].reshape(3, 3)
            Rm2 = _project_to_so3_np(Rm)
            out = out.copy()
            out[:9] = Rm2.reshape(-1)
        return out.astype(np.float32, copy=False)
    
    def save_model(self, filepath, *, verbose: bool = True):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config() if hasattr(self.model, "get_config") else None,
            'scaler_imu': self.scaler_imu,
            'scaler_pose': self.scaler_pose,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        if bool(verbose):
            print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_imu = checkpoint['scaler_imu']
        self.scaler_pose = checkpoint['scaler_pose']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"模型已从 {filepath} 加载")

    @staticmethod
    def from_checkpoint(filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """从 checkpoint 自动恢复模型结构 + 训练器（支持不同 hidden_dims）。"""
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=device)
        cfg = checkpoint.get("model_config", None)
        model = None
        if isinstance(cfg, dict):
            cls_name = str(cfg.get("class", "IMUCameraMLP"))
            if cls_name == "IMUCameraMLP":
                model = IMUCameraMLP(
                    input_dim=int(cfg.get("input_dim", 17)),
                    hidden_dims=list(cfg.get("hidden_dims", [256, 256, 128])),
                    output_dim=int(cfg.get("output_dim", 12)),
                    dropout=float(cfg.get("dropout", 0.2)),
                )
            elif cls_name == "IMUCameraPoseMLP6D":
                model = IMUCameraPoseMLP6D(
                    input_dim=int(cfg.get("input_dim", 17)),
                    hidden_dims=list(cfg.get("hidden_dims", [256, 256, 128])),
                    dropout=float(cfg.get("dropout", 0.2)),
                )
            elif cls_name == "IMUCameraPoseGRU6D":
                model = IMUCameraPoseGRU6D(
                    input_dim=int(cfg.get("input_dim", 17)),
                    embed_dim=int(cfg.get("embed_dim", 128)),
                    gru_hidden_dim=int(cfg.get("gru_hidden_dim", 256)),
                    gru_num_layers=int(cfg.get("gru_num_layers", 2)),
                    head_dim=int(cfg.get("head_dim", 256)),
                    head_blocks=int(cfg.get("head_blocks", 2)),
                    dropout=float(cfg.get("dropout", 0.2)),
                )
        if model is None:
            model = IMUCameraMLP(input_dim=17, hidden_dims=[256, 256, 128], output_dim=12)

        trainer = IMUCameraMLPTrainer(model, device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.scaler_imu = checkpoint.get('scaler_imu', trainer.scaler_imu)
        trainer.scaler_pose = checkpoint.get('scaler_pose', trainer.scaler_pose)
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        return trainer
    
    def plot_training_history(self, save_path=None, *, show: bool = True, tight_layout: bool = False):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练历史')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失 (对数)')
        plt.title('训练历史 (对数)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        # 注意：部分 Windows 环境下 numpy.linalg.inv 可能触发底层崩溃，
        # 而 matplotlib.tight_layout 会间接调用 inv；默认关闭 tight_layout 以规避。
        if bool(tight_layout):
            plt.tight_layout()
        
        if save_path:
            if bool(tight_layout):
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.savefig(save_path, dpi=300)
            print(f"训练历史图已保存到: {save_path}")

        if bool(show):
            plt.show()
        else:
            plt.close()

class DataCollector:
    """数据收集器 - 从标定结果中收集IMU和相机位姿数据"""
    
    def __init__(self):
        self.imu_data_list = []
        self.camera_poses_list = []
    
    def load_calibration_data(self, calibration_dir="calibration_results"):
        """从标定结果目录加载数据"""
        print("正在加载标定数据...")
        
        # 查找最新的标定结果
        calibration_files = glob.glob(os.path.join(calibration_dir, "camera_calibration_*.json"))
        if not calibration_files:
            raise FileNotFoundError(f"在 {calibration_dir} 中未找到标定结果文件")
        
        latest_file = max(calibration_files, key=os.path.getctime)
        print(f"加载标定文件: {latest_file}")
        
        with open(latest_file, 'r') as f:
            calib_data = json.load(f)
        
        # 查找对应的IMU数据文件
        imu_files = glob.glob(os.path.join(calibration_dir, "imu_to_camera_extrinsics_*.json"))
        if not imu_files:
            raise FileNotFoundError(f"在 {calibration_dir} 中未找到IMU外参文件")
        
        latest_imu_file = max(imu_files, key=os.path.getctime)
        print(f"加载IMU外参文件: {latest_imu_file}")
        
        with open(latest_imu_file, 'r') as f:
            imu_extrinsics = json.load(f)
        
        return calib_data, imu_extrinsics
    
    def load_imu_data_from_photos(self, photos_dir="captured_photos"):
        """从拍摄的照片目录中加载IMU数据"""
        print("正在从照片目录加载IMU数据...")
        
        # 这里需要根据实际的照片命名规则来加载IMU数据
        # 假设IMU数据保存在照片的元数据中或者单独的文件中
        imu_data_files = glob.glob(os.path.join(photos_dir, "*.json"))
        
        imu_data_list = []
        for file in sorted(imu_data_files):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if 'imu_data' in data:
                        imu_data_list.append(data['imu_data'])
            except Exception as e:
                print(f"加载IMU数据文件 {file} 失败: {e}")
        
        return imu_data_list
    
    def prepare_training_data(self, calibration_dir="calibration_results", photos_dir="captured_photos"):
        """准备训练数据"""
        print("准备训练数据...")
        
        # 加载标定数据
        calib_data, imu_extrinsics = self.load_calibration_data(calibration_dir)
        
        # 获取相机位姿
        rvecs = np.array(calib_data['extrinsics']['rvecs'])
        tvecs = np.array(calib_data['extrinsics']['tvecs'])
        
        # 转换为旋转矩阵
        camera_poses = []
        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rvec)
            # 将旋转矩阵和平移向量展平
            pose = np.concatenate([R.flatten(), tvec.flatten()])
            camera_poses.append(pose)
        
        camera_poses = np.array(camera_poses)
        
        # 加载IMU数据
        imu_data_list = self.load_imu_data_from_photos(photos_dir)
        
        if len(imu_data_list) != len(camera_poses):
            print(f"警告: IMU数据数量 ({len(imu_data_list)}) 与相机位姿数量 ({len(camera_poses)}) 不匹配")
            min_len = min(len(imu_data_list), len(camera_poses))
            imu_data_list = imu_data_list[:min_len]
            camera_poses = camera_poses[:min_len]
        
        # 转换IMU数据为numpy数组
        imu_features = []
        for imu_data in imu_data_list:
            if isinstance(imu_data, dict):
                # 提取所有IMU特征
                features = []
                features.extend(imu_data.get('acc', [0, 0, 0]))
                features.extend(imu_data.get('gyro', [0, 0, 0]))
                features.extend(imu_data.get('angle', [0, 0, 0]))
                features.extend(imu_data.get('mag', [0, 0, 0]))
                features.append(imu_data.get('pressure', 0))
                features.append(imu_data.get('altitude', 0))
                features.extend(imu_data.get('quaternion', [0, 0, 0, 0]))
                
                # 确保有17个特征
                if len(features) < 17:
                    features.extend([0] * (17 - len(features)))
                elif len(features) > 17:
                    features = features[:17]
                
                imu_features.append(features)
            else:
                print(f"警告: 无效的IMU数据格式: {imu_data}")
        
        imu_features = np.array(imu_features)
        
        print(f"数据准备完成:")
        print(f"  IMU特征形状: {imu_features.shape}")
        print(f"  相机位姿形状: {camera_poses.shape}")
        
        return imu_features, camera_poses

def _project_to_so3_np(Rm: np.ndarray) -> np.ndarray:
    """将 3x3 矩阵投影到旋转矩阵（Gram-Schmidt），避免 numpy.linalg.svd 触发底层崩溃。"""
    A = np.asarray(Rm, dtype=np.float64).reshape(3, 3)
    a1 = A[:, 0]
    a2 = A[:, 1]
    n1 = float(np.sqrt(np.sum(a1 * a1)))
    if (not np.isfinite(n1)) or n1 < 1e-12:
        return np.eye(3, dtype=np.float64)
    b1 = a1 / n1
    a2o = a2 - float(np.sum(b1 * a2)) * b1
    n2 = float(np.sqrt(np.sum(a2o * a2o)))
    if (not np.isfinite(n2)) or n2 < 1e-12:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.sum(tmp * b1))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        a2o = tmp - float(np.sum(b1 * tmp)) * b1
        n2 = float(np.sqrt(np.sum(a2o * a2o)))
        if n2 < 1e-12:
            return np.eye(3, dtype=np.float64)
    b2 = a2o / n2
    b3 = np.cross(b1, b2)
    Rm2 = np.stack([b1, b2, b3], axis=1)
    return Rm2


def evaluate_model_accuracy(
    model,
    test_dataset,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    *,
    scaler_pose=None,
    batch_size: int = 256,
    project_to_so3: bool = True,
):
    """评估模型精度（默认在反标准化后的真实尺度上计算）。"""
    model.eval()

    rotation_errors: list[float] = []
    translation_errors: list[float] = []

    with torch.no_grad():
        loader = DataLoader(test_dataset, batch_size=int(max(1, batch_size)), shuffle=False)
        pred_list = []
        true_list = []
        for imu_data, poses in loader:
            imu_data = imu_data.to(device)
            poses = poses.to(device)
            pred = model(imu_data)
            pred_list.append(pred.detach().cpu().numpy())
            true_list.append(poses.detach().cpu().numpy())

        if not pred_list:
            print("评估失败：测试集为空")
            return np.array([]), np.array([])

        pred_np = np.concatenate(pred_list, axis=0)
        true_np = np.concatenate(true_list, axis=0)

        # 反标准化（否则旋转/平移误差没有物理意义）
        if scaler_pose is not None:
            try:
                pred_np = scaler_pose.inverse_transform(pred_np)
                true_np = scaler_pose.inverse_transform(true_np)
            except Exception as e:
                print(f"警告: scaler_pose.inverse_transform 失败，将在标准化空间评估 err={e}")

        for k in range(pred_np.shape[0]):
            pred_R = pred_np[k, :9].reshape(3, 3)
            true_R = true_np[k, :9].reshape(3, 3)
            if bool(project_to_so3):
                pred_R = _project_to_so3_np(pred_R)
                true_R = _project_to_so3_np(true_R)
            # 避免 numpy 矩阵乘法（部分环境下会触发底层崩溃）
            trc = float(np.sum(pred_R * true_R))  # trace(pred_R.T @ true_R)
            cos_theta = (trc - 1.0) / 2.0
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
            rotation_errors.append(float(np.arccos(cos_theta)))

            pred_t = pred_np[k, 9:12]
            true_t = true_np[k, 9:12]
            d = pred_t - true_t
            translation_errors.append(float(np.sqrt(np.sum(d * d))))

    rotation_errors_np = np.array(rotation_errors, dtype=np.float64)
    translation_errors_np = np.array(translation_errors, dtype=np.float64)

    print(f"模型评估结果:")
    if rotation_errors_np.size > 0:
        print(f"  旋转误差: {np.mean(rotation_errors_np):.4f} ± {np.std(rotation_errors_np):.4f} 弧度")
        print(f"  旋转误差: {np.degrees(np.mean(rotation_errors_np)):.2f} ± {np.degrees(np.std(rotation_errors_np)):.2f} 度")
    if translation_errors_np.size > 0:
        print(f"  平移误差: {np.mean(translation_errors_np):.4f} ± {np.std(translation_errors_np):.4f} 单位")

    return rotation_errors_np, translation_errors_np

def main():
    """主函数"""
    print("="*80)
    print("IMU到相机位姿MLP网络训练")
    print("="*80)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建数据收集器
    collector = DataCollector()
    
    try:
        # 准备训练数据
        imu_data, camera_poses = collector.prepare_training_data()
        
        if len(imu_data) < 10:
            print(f"错误: 数据量不足 ({len(imu_data)} 个样本)，至少需要10个样本")
            return
        
        # 创建模型
        model = IMUCameraMLP(input_dim=17, hidden_dims=[256, 256, 128], output_dim=12)
        trainer = IMUCameraMLPTrainer(model, device)
        
        # 准备数据集
        train_dataset, test_dataset = trainer.prepare_data(imu_data, camera_poses)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 训练模型
        best_loss = trainer.train(train_dataset, test_dataset, epochs=1000, batch_size=16)
        
        # 绘制训练历史
        trainer.plot_training_history('training_history.png')
        
        # 评估模型
        print("\n评估模型性能...")
        rotation_errors, translation_errors = evaluate_model_accuracy(
            model,
            test_dataset,
            device,
            scaler_pose=trainer.scaler_pose,
            batch_size=256,
            project_to_so3=True,
        )
        
        # 保存模型
        model_path = f"imu_camera_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        trainer.save_model(model_path)
        
        print(f"\n训练完成！")
        print(f"模型已保存到: {model_path}")
        print(f"最佳验证损失: {best_loss:.6f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
