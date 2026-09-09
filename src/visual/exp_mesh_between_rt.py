#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Exp_Mehsh_Between_RT.py

目标：
  读取序列图像 -> 每帧重建 3D mesh_t -> 计算帧间 mesh 的刚体配准 RT（旋转+平移）

核心输出：
  - pairwise：T_{t-1 <- t}  (把 frame t 的点变换到 frame t-1 坐标系)
  - cumulative：T_{0 <- t}  (把 frame t 的点变换到 frame 0 坐标系)

重要说明（务必看）：
  - 本脚本的 mesh 是“伪3D高度场”(u,v,h)：
      x=u(像素), y=v(像素), z=h(u,v)(由阈值/梯度/强度构造)
    因此这里的 RT 更准确叫“形状对齐变换(Shape-Alignment RT)”，不等价于物理相机位姿。
  - 为避免混淆，脚本会额外保存 shape_rt_pairwise_4x4.txt / shape_rt_cumulative_4x4.txt，
    并在 meta.json 中标注 coordinate_system=uvh_heightfield。

典型方案（本脚本都支持）：
  A) Tracking ICP（推荐优先用，速度快）：
     - 初值用上一帧的结果（累计变换）作为 init
     - 直接 ICP refine 得到当前帧 RT

  B) Global + ICP（当初值不可靠/漂移大时）：
     - 对下采样点云计算 FPFH
     - RANSAC 粗配准得到 init
     - 再 ICP refine

依赖：
  - open3d
  - opencv-python (cv2)
  - numpy
  - 你现有的 ImageProcessor（来自 ChessBoard_Cal_RT_CMP_3DICP_2DFFT_Fea_Matching_MIS_Op3Test3.py）

环境变量（常用）：
  INPUT_DIR   : 图像目录（默认 D:\\reloc3r\\Data_IMU_Camera_Pose_3\\captured_photos）
  OUTPUT_DIR  : 输出目录（默认 D:\\reloc3r\\Data_IMU_Camera_Pose_3\\Ours_Camera_Pose_60f_eval_vis）
  MAX_FRAMES  : 最多处理帧数（默认 0=不限制）
  FRAME_STEP  : 帧间隔（默认 1）

  SAMPLE_POINTS        : 每个 mesh 采样点数（默认 60000；0 表示直接用顶点）
  VOXEL_SIZE           : 点云下采样 voxel（默认 2.0；<=0 表示不下采样）
  ICP_METHOD           : point_to_plane / point_to_point（默认 point_to_plane）
  ICP_MAX_CORR         : ICP 最大对应距离（默认 10.0）
  ICP_MAX_ITER         : ICP 最大迭代（默认 80）

  USE_GLOBAL_RANSAC     : 1 启用全局粗配准（默认 0）
  RANSAC_VOXEL_SIZE     : RANSAC 用 voxel（默认 5.0）
  RANSAC_MAX_CORR_MULT  : RANSAC 最大对应距离=mult*voxel（默认 2.5）
  RANSAC_MAX_ITER       : RANSAC 最大迭代（默认 50000）
  RANSAC_CONFIDENCE     : RANSAC 置信度（默认 0.999）

  SAVE_MESH_ALIGNED     : 1 保存对齐后的 mesh（默认 0）
  SAVE_PAIR_RESULTS     : 1 保存帧间匹配结果（默认 1）
  PRINT_RT              : 1 在控制台打印RT矩阵（默认 1）
  PRINT_RT_EVERY        : 每隔N帧打印一次（默认 1；比如 10 表示每10帧打印）
  PRINT_RT_MODE         : pairwise / cumulative / both（默认 pairwise）
  VIS_OPEN3D            : 1 弹出 Open3D 可视化（默认 0）
  VIS_SHOW_RT_FRAMES    : 1 绘制每帧坐标系/轨迹（默认 1，当 VIS_OPEN3D=1 时生效）
  VIS_RT_EVERY          : 每隔N帧绘制一个坐标系（默认 1）
  VIS_FRAME_SIZE        : 坐标系大小（默认 auto；也可直接给数值）
  VIS_SHOW_CAMERA_FRUSTUM : 1 绘制相机视锥（默认 1，当 VIS_OPEN3D=1 时生效）
  VIS_TRAJ_USE_ALL_FRAMES  : 1 用全部帧绘制轨迹（默认 1；否则只画 VIS_MAX_MESHES 范围内）

  POSE_INTERPRETATION      : camera / object（默认 camera）
    - camera：假设“每帧 mesh_t 在该帧相机坐标系里重建”，则累计矩阵 T_{0<-t} 可视为相机位姿(c2w)
    - object：假设“相机不动、物体在动”，则把相机位姿当作 inverse(T_{0<-t})

匹配增强（新增，默认已偏“稳”）：
  ICP_USE_COLORED           : 1 启用 Colored-ICP refine（默认 1，若点云无颜色会自动跳过）
  ICP_COLORED_LAMBDA_GEOM   : Colored-ICP 的几何权重（默认 0.968）
  POSE_GRAPH_OPTIMIZE       : 1 在序列结束后做 PoseGraph 全局优化（默认 1）
  POSE_GRAPH_VOXEL          : PoseGraph 用的更强下采样 voxel（默认 6.0）
  POSE_GRAPH_MAX_CORR_MULT  : PoseGraph 信息矩阵 max_corr = mult*voxel（默认 2.5）
"""

import os
import sys
import json
import time
import math
import csv
from dataclasses import dataclass
from typing import Any
import re

import numpy as np
import cv2
import open3d as o3d

import ChessBoard_Cal_RT_CMP_3DICP_2DFFT_Fea_Matching_MIS_Op3Test3 as cb


ImageProcessor = cb.ImageProcessor


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _parse_size_token(tok: str) -> tuple[int, int] | None:
    """
    解析棋盘格 inner-corners 尺寸：
      - 支持 "9x8" / "9,8" / "9 8"
    返回 (cols, rows)。
    """
    try:
        s = str(tok).strip().lower().replace("×", "x")
        for sep in ("x", ",", " "):
            if sep in s:
                a, b = [p.strip() for p in s.split(sep) if p.strip()]
                if len(a) == 0 or len(b) == 0:
                    return None
                c = int(a)
                r = int(b)
                if c > 1 and r > 1:
                    return (c, r)
        return None
    except Exception:
        return None


def _parse_size_list(env_name: str, default: str) -> list[tuple[int, int]]:
    """
    从环境变量读取候选棋盘格尺寸列表：
      - 形如 "9x8;8x7" 或 "9x8,8x7"
    """
    s = os.environ.get(env_name, default)
    parts = str(s).replace(",", ";").split(";")
    out: list[tuple[int, int]] = []
    for p in parts:
        t = _parse_size_token(p)
        if t is not None:
            out.append(t)
    # 去重
    uniq: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def _parse_square_size_map(env_name: str, default: str) -> dict[tuple[int, int], float]:
    """
    为不同棋盘格 inner-corners 尺寸指定不同的 square_size（单位同 CHESSBOARD_SQUARE_SIZE）：
      - 例如： "9x8=8;8x7=8;8x5=5"
    返回：{(cols, rows): square_size}
    """
    s = os.environ.get(env_name, default)
    out: dict[tuple[int, int], float] = {}
    for part in str(s).replace(",", ";").split(";"):
        part = part.strip()
        if not part or ("=" not in part):
            continue
        k, v = [p.strip() for p in part.split("=", 1)]
        sz = _parse_size_token(k)
        if sz is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        if np.isfinite(f) and f > 0:
            out[sz] = float(f)
    return out


def _detect_chessboard_corners(
    gray: np.ndarray,
    *,
    try_sizes: list[tuple[int, int]],
    refine: bool = True,
) -> tuple[bool, np.ndarray | None, tuple[int, int] | None]:
    """
    棋盘格角点检测（inner corners）：
      - 优先用 findChessboardCornersSB（更稳，OpenCV>=4.5 常见）
      - 兜底用 findChessboardCorners + cornerSubPix
    返回 (ok, corners(N,1,2), used_size)。
    """
    if gray is None:
        return False, None, None
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    flags_sb = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE
    used = None
    corners = None
    ok = False
    for sz in try_sizes:
        try:
            if hasattr(cv2, "findChessboardCornersSB"):
                ok0, c0 = cv2.findChessboardCornersSB(gray, sz, flags_sb)
            else:
                ok0, c0 = cv2.findChessboardCorners(gray, sz, flags)
            if ok0 and c0 is not None and int(c0.shape[0]) == int(sz[0] * sz[1]):
                ok = True
                corners = c0
                used = sz
                break
        except Exception:
            continue
    if (not ok) or (corners is None) or (used is None):
        return False, None, None
    if refine and (not hasattr(cv2, "findChessboardCornersSB")):
        # SB 通常已经较精细；普通 findChessboardCorners 建议 subpix
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        except Exception:
            pass
    return True, corners, used


def _make_chessboard_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """
    生成棋盘格 3D 点（世界坐标系）：
      - Z=0 的平面
      - 单位由 square_size 决定（例如 8.0 表示 mm）
    返回 shape (N,3) float32。
    """
    objp = np.zeros((int(cols * rows), 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp[:, :2] = grid * float(square_size)
    return objp


def _rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _rot_angle_deg(R: np.ndarray) -> float:
    """从旋转矩阵估计旋转角（0~180deg）。"""
    try:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tr = float(np.trace(R))
        c = (tr - 1.0) * 0.5
        c = float(max(-1.0, min(1.0, c)))
        return float(math.degrees(math.acos(c)))
    except Exception:
        return float("nan")


def _load_intrinsics_json(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        K = np.asarray(j.get("K", None), dtype=np.float64)
        dist = np.asarray(j.get("dist", None), dtype=np.float64).reshape(-1)
        if K.shape != (3, 3):
            return None
        return K, dist
    except Exception:
        return None


def _save_intrinsics_json(path: str, K: np.ndarray, dist: np.ndarray, extra: dict[str, Any] | None = None) -> None:
    try:
        j: dict[str, Any] = {
            "K": np.asarray(K, dtype=np.float64).tolist(),
            "dist": np.asarray(dist, dtype=np.float64).reshape(-1).tolist(),
        }
        if extra:
            j.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(j, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _safe_open_for_write(path: str, *, newline: str | None = None):
    """
    Windows 下常见：文件被 Excel 等占用 => PermissionError。
    兜底策略：写不了就自动换成带时间戳的文件名继续写，避免脚本中断。
    返回 (file_handle, actual_path)。
    """
    try:
        if newline is None:
            f = open(path, "w", encoding="utf-8")
        else:
            f = open(path, "w", encoding="utf-8", newline=newline)
        return f, path
    except PermissionError:
        base, ext = os.path.splitext(path)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        alt = f"{base}_{ts}{ext or '.txt'}"
        if newline is None:
            f = open(alt, "w", encoding="utf-8")
        else:
            f = open(alt, "w", encoding="utf-8", newline=newline)
        print(f"[Warning] cannot write '{path}' (permission denied). Writing to: {alt}", flush=True)
        return f, alt


def _as_bool(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Write bytes to path atomically (best-effort)."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


def _save_json_atomic(path: str, obj: Any) -> None:
    try:
        data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
        _atomic_write_bytes(path, data)
    except Exception:
        # last resort: non-atomic
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def _load_json(path: str) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_progress_rt_npz(path: str, frames: list[int], pairwise: list[np.ndarray], cumulative: list[np.ndarray]) -> None:
    """Persist processed RTs for resume. frames/pairwise/cumulative must have same length."""
    try:
        n = int(len(frames))
        if n == 0:
            return
        F = np.asarray(frames, dtype=np.int32).reshape(-1)
        Pw = np.stack([np.asarray(m, dtype=np.float64).reshape(4, 4) for m in pairwise], axis=0)
        Cu = np.stack([np.asarray(m, dtype=np.float64).reshape(4, 4) for m in cumulative], axis=0)
        tmp = f"{path}.tmp"
        np.savez_compressed(tmp, frames=F, pairwise=Pw, cumulative=Cu)
        # numpy will append .npz if not present; normalize
        if not tmp.lower().endswith(".npz"):
            tmp2 = tmp + ".npz"
        else:
            tmp2 = tmp
        os.replace(tmp2, path)
    except Exception:
        pass


def _load_progress_rt_npz(path: str) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    try:
        z = np.load(path, allow_pickle=False)
        frames = np.asarray(z["frames"], dtype=np.int32).reshape(-1)
        pairwise = np.asarray(z["pairwise"], dtype=np.float64).reshape(-1, 4, 4)
        cumulative = np.asarray(z["cumulative"], dtype=np.float64).reshape(-1, 4, 4)
        n = int(frames.shape[0])
        out_f = [int(x) for x in frames.tolist()]
        out_p = [pairwise[i].copy() for i in range(n)]
        out_c = [cumulative[i].copy() for i in range(n)]
        return out_f, out_p, out_c
    except Exception:
        return [], [], []


def _infer_last_frame_from_vis(vis_dir: str) -> int | None:
    """Fallback: infer last processed cur frame from vis filenames."""
    try:
        if not os.path.isdir(vis_dir):
            return None
        rx = re.compile(r"pair_prev_\d{6}_cur_(\d{6})_cur_in_prev\.ply$", re.IGNORECASE)
        best: int | None = None
        for name in os.listdir(vis_dir):
            m = rx.search(str(name))
            if not m:
                continue
            v = int(m.group(1))
            if best is None or v > best:
                best = v
        return best
    except Exception:
        return None


def _filter_pcd_by_z(pcd: o3d.geometry.PointCloud, *, z_eps: float, min_points: int = 200) -> o3d.geometry.PointCloud:
    """过滤掉 z<=eps 的点（高度场背景/无效区域），避免 ICP 被大平面拖偏。"""
    try:
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size == 0:
            return pcd
        z = pts[:, 2]
        m = np.isfinite(z) & (z > float(z_eps))
        if int(np.count_nonzero(m)) < int(min_points):
            return pcd
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts[m])
        if pcd.has_normals():
            n = np.asarray(pcd.normals, dtype=np.float64)
            if n.shape[0] == pts.shape[0]:
                out.normals = o3d.utility.Vector3dVector(n[m])
        if pcd.has_colors():
            c = np.asarray(pcd.colors, dtype=np.float64)
            if c.shape[0] == pts.shape[0]:
                out.colors = o3d.utility.Vector3dVector(c[m])
        return out
    except Exception:
        return pcd


def _filter_pcd_by_xy_bbox(
    pcd: o3d.geometry.PointCloud,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    margin: float,
    min_points: int = 200,
) -> o3d.geometry.PointCloud:
    """用 xy bbox (可加 margin) 裁剪点云，抑制全图背景对 ICP 的干扰。"""
    try:
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size == 0:
            return pcd
        x0 = float(min_x) - float(margin)
        y0 = float(min_y) - float(margin)
        x1 = float(max_x) + float(margin)
        y1 = float(max_y) + float(margin)
        m = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        if int(np.count_nonzero(m)) < int(min_points):
            return pcd
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts[m])
        if pcd.has_normals():
            n = np.asarray(pcd.normals, dtype=np.float64)
            if n.shape[0] == pts.shape[0]:
                out.normals = o3d.utility.Vector3dVector(n[m])
        if pcd.has_colors():
            c = np.asarray(pcd.colors, dtype=np.float64)
            if c.shape[0] == pts.shape[0]:
                out.colors = o3d.utility.Vector3dVector(c[m])
        return out
    except Exception:
        return pcd


def _pcd_len(pcd: o3d.geometry.PointCloud | None) -> int:
    try:
        if pcd is None:
            return 0
        return int(len(pcd.points))
    except Exception:
        return 0


def _pcd_add_height_colors(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    给点云添加颜色（来自 z 高度归一化），用于 Colored-ICP：
      - 颜色与光照无关（比直接用图像灰度稳）
    """
    try:
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size == 0:
            return pcd
        z = pts[:, 2].astype(np.float64)
        m = np.isfinite(z)
        if int(np.count_nonzero(m)) < 10:
            return pcd
        z2 = z[m]
        # robust min/max by percentiles
        lo = float(np.percentile(z2, 2.0))
        hi = float(np.percentile(z2, 98.0))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo + 1e-9:
            lo = float(np.nanmin(z2))
            hi = float(np.nanmax(z2))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo + 1e-9:
            return pcd
        t = (z - lo) / (hi - lo)
        t = np.clip(t, 0.0, 1.0)
        col = np.stack([t, t, t], axis=1).astype(np.float64)
        out = o3d.geometry.PointCloud(pcd)
        out.colors = o3d.utility.Vector3dVector(col)
        return out
    except Exception:
        return pcd


def _estimate_rigid_T_svd(src: np.ndarray, tgt: np.ndarray) -> np.ndarray | None:
    """
    估计刚体变换 T_{tgt<-src}（SVD/Kabsch）：
      - src/tgt: (N,3) 对应点
      - 返回 4x4；失败返回 None
    """
    try:
        A = np.asarray(src, dtype=np.float64).reshape(-1, 3)
        B = np.asarray(tgt, dtype=np.float64).reshape(-1, 3)
        if A.shape[0] < 3 or B.shape[0] != A.shape[0]:
            return None
        ma = A.mean(axis=0)
        mb = B.mean(axis=0)
        AA = A - ma
        BB = B - mb
        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = (Vt.T @ U.T)
        # reflection fix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = (Vt.T @ U.T)
        t = (mb - R @ ma).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        if not np.isfinite(T).all():
            return None
        return T
    except Exception:
        return None


def _euler_xyz_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """右手系，先绕X再绕Y再绕Z（XYZ）"""
    rx = math.radians(float(rx_deg))
    ry = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return (Rz @ Ry @ Rx).astype(np.float64)


def _make_delta_T(rx_deg: float, ry_deg: float, rz_deg: float, tx: float, ty: float, tz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _euler_xyz_to_R(rx_deg, ry_deg, rz_deg)
    T[:3, 3] = np.array([float(tx), float(ty), float(tz)], dtype=np.float64)
    return T


def _parse_float_list(env: str, default_csv: str) -> list[float]:
    s = os.environ.get(env, default_csv)
    out: list[float] = []
    for part in str(s).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            pass
    return out


def _safe_get(stats: dict[str, float] | None, k: str, default: float = float("nan")) -> float:
    try:
        if not stats:
            return float(default)
        return float(stats.get(k, default))
    except Exception:
        return float(default)


def _motion_gate(prev_stats: dict[str, float] | None, cur_stats: dict[str, float] | None) -> dict[str, float | str | int]:
    """
    运动/跳变判据（用于选择恢复策略）：
      - 小位移：走 tracking（必要时小范围 greedy）
      - 中位移：走“绕 nz_center 的球面 yaw/pitch 采样 + 短 ICP”
      - 跳变：nz_ratio / nz_bbox / nz_center 明显突变 => 直接 keyframe / 兜底（RANSAC 仅在此启用）
    """
    # thresholds (tuned for pixel-space uv)
    d_small = float(os.environ.get("NZ_CENTER_D_SMALL", "25"))
    d_med = float(os.environ.get("NZ_CENTER_D_MED", "120"))
    d_jump = float(os.environ.get("NZ_CENTER_D_JUMP", "200"))
    ratio_jump_mult = float(os.environ.get("NZ_RATIO_JUMP_MULT", "2.2"))
    bbox_shift_jump = float(os.environ.get("NZ_BBOX_SHIFT_JUMP", "120"))

    px = _safe_get(prev_stats, "nz_center_x")
    py = _safe_get(prev_stats, "nz_center_y")
    cx = _safe_get(cur_stats, "nz_center_x")
    cy = _safe_get(cur_stats, "nz_center_y")
    dxy = float("inf")
    if np.isfinite(px) and np.isfinite(py) and np.isfinite(cx) and np.isfinite(cy):
        dxy = float(np.linalg.norm(np.array([px - cx, py - cy], dtype=np.float64)))

    pr = _safe_get(prev_stats, "nz_ratio", 0.0)
    cr = _safe_get(cur_stats, "nz_ratio", 0.0)
    # ratio multiplier: max(cr/pr, pr/cr)
    ratio_mult = 1.0
    if pr > 1e-12 and cr > 1e-12:
        ratio_mult = float(max(cr / pr, pr / cr))
    elif (pr <= 1e-12 and cr > 1e-12) or (cr <= 1e-12 and pr > 1e-12):
        ratio_mult = float("inf")

    # bbox shift: max abs delta of nz bbox (xy)
    pmx = _safe_get(prev_stats, "nz_bbox_min_x")
    pmy = _safe_get(prev_stats, "nz_bbox_min_y")
    pMx = _safe_get(prev_stats, "nz_bbox_max_x")
    pMy = _safe_get(prev_stats, "nz_bbox_max_y")
    cmx = _safe_get(cur_stats, "nz_bbox_min_x")
    cmy = _safe_get(cur_stats, "nz_bbox_min_y")
    cMx = _safe_get(cur_stats, "nz_bbox_max_x")
    cMy = _safe_get(cur_stats, "nz_bbox_max_y")
    bbox_shift = 0.0
    if all(np.isfinite(v) for v in [pmx, pmy, pMx, pMy, cmx, cmy, cMx, cMy]):
        bbox_shift = float(
            max(
                abs(pmx - cmx),
                abs(pmy - cmy),
                abs(pMx - cMx),
                abs(pMy - cMy),
            )
        )

    jump = int((dxy > d_jump) or (ratio_mult >= ratio_jump_mult) or (bbox_shift >= bbox_shift_jump))
    if jump:
        mode = "jump"
    elif dxy <= d_small:
        mode = "small"
    elif dxy <= d_med:
        mode = "medium"
    else:
        # large but not flagged as jump by ratio/bbox, still treat as medium (use sphere sampling)
        mode = "medium"

    return {
        "nz_center_dxy": float(dxy),
        "nz_ratio_mult": float(ratio_mult),
        "nz_bbox_shift": float(bbox_shift),
        "gate_jump": int(jump),
        "gate_mode": str(mode),
    }


def _sphere_pose_candidates(
    *,
    init_T: np.ndarray,
    prev_stats: dict[str, float] | None,
    cur_stats: dict[str, float] | None,
) -> list[tuple[str, np.ndarray]]:
    """
    绕有效区域中心(nz_center)做“球面姿态空间”采样（yaw/pitch），用于生成更强的恢复候选初值。
    核心：旋转绕目标中心，平移由 t = c_prev - R*c_cur 保证中心先对齐，再加少量 xy 平移扰动。
    """
    if not _as_bool(os.environ.get("USE_SPHERE_SAMPLING", "1")):
        return []
    if prev_stats is None or cur_stats is None:
        return []
    px = _safe_get(prev_stats, "nz_center_x")
    py = _safe_get(prev_stats, "nz_center_y")
    pz = _safe_get(prev_stats, "nz_center_z", 0.0)
    cx = _safe_get(cur_stats, "nz_center_x")
    cy = _safe_get(cur_stats, "nz_center_y")
    cz = _safe_get(cur_stats, "nz_center_z", 0.0)
    if not all(np.isfinite(v) for v in [px, py, pz, cx, cy, cz]):
        return []

    c_prev = np.array([px, py, pz], dtype=np.float64)
    c_cur = np.array([cx, cy, cz], dtype=np.float64)

    yaw_list = _parse_float_list("SPHERE_YAW_DEG_LIST", "0,10,-10,20,-20,30,-30")
    pitch_list = _parse_float_list("SPHERE_PITCH_DEG_LIST", "0,6,-6,12,-12")
    trans_list = _parse_float_list("SPHERE_TRANS_XY_LIST", "0,5,-5,10,-10,20,-20")
    max_cands = int(os.environ.get("SPHERE_MAX_CANDS", "64"))

    T0 = np.asarray(init_T, dtype=np.float64)
    if T0.shape != (4, 4):
        T0 = np.eye(4, dtype=np.float64)
    R0 = T0[:3, :3].copy()

    cands: list[tuple[str, np.ndarray]] = []
    # always include the raw init
    cands.append(("init", T0))

    # generate candidates
    for yaw in yaw_list:
        for pitch in pitch_list:
            dR = _euler_xyz_to_R(float(pitch), 0.0, float(yaw))
            R = (dR @ R0).astype(np.float64)
            t_center = (c_prev - R @ c_cur).astype(np.float64)
            # single-axis xy translations (keep combos limited by max_cands)
            for t in trans_list:
                if abs(float(t)) < 1e-9:
                    # keep only once
                    Tx = np.eye(4, dtype=np.float64)
                    Tx[:3, :3] = R
                    Tx[:3, 3] = t_center
                    cands.append((f"sphere_yaw{yaw:+g}_pit{pitch:+g}", Tx))
                    continue
                for axis in ("x", "y"):
                    dt = np.array([float(t), 0.0, 0.0], dtype=np.float64) if axis == "x" else np.array([0.0, float(t), 0.0], dtype=np.float64)
                    Tx = np.eye(4, dtype=np.float64)
                    Tx[:3, :3] = R
                    Tx[:3, 3] = t_center + dt
                    cands.append((f"sphere_yaw{yaw:+g}_pit{pitch:+g}_d{axis}{t:+g}", Tx))
                if len(cands) >= max(8, max_cands * 2):
                    break
            if len(cands) >= max(8, max_cands * 2):
                break
        if len(cands) >= max(8, max_cands * 2):
            break

    # de-dup by name and cap
    seen: set[str] = set()
    uniq: list[tuple[str, np.ndarray]] = []
    for name, T in cands:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, T))
    return uniq[: max(1, max_cands)]


def _greedy_candidates(init_T: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    贪婪多初值：围绕 init_T 做小范围 6DoF 扰动，返回候选列表。
    约定：T_candidate = Δ @ init_T （在目标坐标系下扰动 init）
    """
    # 默认加大搜索范围（你也可用环境变量覆盖）
    rot_list = _parse_float_list("GREEDY_ROT_DEG_LIST", "0,5,-5,10,-10,15,-15,20,-20")
    trans_list = _parse_float_list("GREEDY_TRANS_LIST", "0,3,-3,6,-6,10,-10,15,-15")
    # 只做少量组合，避免指数爆炸：优先单轴扰动 + 少量组合
    cands: list[tuple[str, np.ndarray]] = [("init", np.asarray(init_T, dtype=np.float64))]
    # single-axis rotations
    for a in rot_list:
        if abs(a) < 1e-9:
            continue
        for axis in ("x", "y", "z"):
            rx, ry, rz = (a, 0.0, 0.0) if axis == "x" else (0.0, a, 0.0) if axis == "y" else (0.0, 0.0, a)
            dT = _make_delta_T(rx, ry, rz, 0.0, 0.0, 0.0)
            cands.append((f"rot_{axis}_{a:+g}", dT @ init_T))
    # single-axis translations
    for t in trans_list:
        if abs(t) < 1e-9:
            continue
        for axis in ("x", "y", "z"):
            tx, ty, tz = (t, 0.0, 0.0) if axis == "x" else (0.0, t, 0.0) if axis == "y" else (0.0, 0.0, t)
            dT = _make_delta_T(0.0, 0.0, 0.0, tx, ty, tz)
            cands.append((f"trans_{axis}_{t:+g}", dT @ init_T))
    # small coupled: yaw + xy shift（取更多组合，但仍受 GREEDY_MAX_CANDS 限制）
    for yaw in [v for v in rot_list if abs(v) > 1e-9][:4]:
        for tx in [v for v in trans_list if abs(v) > 1e-9][:4]:
            for ty in [v for v in trans_list if abs(v) > 1e-9][:4]:
                dT = _make_delta_T(0.0, 0.0, yaw, tx, ty, 0.0)
                cands.append((f"yaw_xy_{yaw:+g}_{tx:+g}_{ty:+g}", dT @ init_T))
    # 去重（按描述去重即可）
    seen: set[str] = set()
    uniq: list[tuple[str, np.ndarray]] = []
    for name, T in cands:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, T))
    # 限制候选数，避免太慢
    max_cands = int(os.environ.get("GREEDY_MAX_CANDS", "64"))
    return uniq[: max(1, max_cands)]


def _list_images(dir_path: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    names = [n for n in os.listdir(dir_path) if n.lower().endswith(exts)]
    names.sort()
    return [os.path.join(dir_path, n) for n in names]


def _ensure_mesh_normals(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    return mesh


def _mesh_basic_stats(mesh: o3d.geometry.TriangleMesh) -> dict[str, float]:
    """
    输出两套统计：
    - 全量顶点统计（通常会被“整张图像网格”主导，容易看起来恒定）
    - 有效区域统计：只对 z>0 的顶点统计（更接近 mask/目标区域）
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if v.size == 0:
        return {}
    z = v[:, 2].astype(np.float64)

    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    center = v.mean(axis=0)
    extent = vmax - vmin
    diag = float(np.linalg.norm(extent))
    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    zmean = float(np.nanmean(z))

    # effective region: z > eps
    eps = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
    m = np.isfinite(z) & (z > eps)
    nz_ratio = float(np.count_nonzero(m)) / float(max(1, z.shape[0]))
    out: dict[str, float] = {
        "bbox_min_x": float(vmin[0]),
        "bbox_min_y": float(vmin[1]),
        "bbox_min_z": float(vmin[2]),
        "bbox_max_x": float(vmax[0]),
        "bbox_max_y": float(vmax[1]),
        "bbox_max_z": float(vmax[2]),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "diag": float(diag),
        "z_min": zmin,
        "z_max": zmax,
        "z_mean": zmean,
        "nz_ratio": float(nz_ratio),
    }
    if np.any(m):
        v_nz = v[m]
        z_nz = z[m]
        vmin2 = v_nz.min(axis=0)
        vmax2 = v_nz.max(axis=0)
        center2 = v_nz.mean(axis=0)
        extent2 = vmax2 - vmin2
        diag2 = float(np.linalg.norm(extent2))
        out.update({
            "nz_bbox_min_x": float(vmin2[0]),
            "nz_bbox_min_y": float(vmin2[1]),
            "nz_bbox_min_z": float(vmin2[2]),
            "nz_bbox_max_x": float(vmax2[0]),
            "nz_bbox_max_y": float(vmax2[1]),
            "nz_bbox_max_z": float(vmax2[2]),
            "nz_center_x": float(center2[0]),
            "nz_center_y": float(center2[1]),
            "nz_center_z": float(center2[2]),
            "nz_diag": float(diag2),
            "nz_z_min": float(np.nanmin(z_nz)),
            "nz_z_max": float(np.nanmax(z_nz)),
            "nz_z_mean": float(np.nanmean(z_nz)),
        })
    return out


def _mesh_to_height_maps(
    mesh: o3d.geometry.TriangleMesh,
    *,
    h: int,
    w: int,
    z_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    把 mesh 顶点 (u,v,z) 回填成二维高度图，便于做稠密 2D 匹配回投影。

    关键事实（已在 ImageProcessor._o3d_build_height_mesh 中确认）：
      - 顶点坐标就是像素坐标：x=xx, y=yy（规则网格采样，步长为 step）
      - mask 外区域 z 被写成 0

    返回：
      - h_map: shape (H,W), float32，高度（无值为 NaN）
      - valid: shape (H,W), bool，有效点（z>z_eps）
    """
    H = int(max(1, h))
    W = int(max(1, w))
    h_map = np.full((H, W), np.nan, dtype=np.float32)
    valid = np.zeros((H, W), dtype=np.bool_)
    try:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        if v.size == 0:
            return h_map, valid
        uu = np.rint(v[:, 0]).astype(np.int32)
        vv = np.rint(v[:, 1]).astype(np.int32)
        zz = v[:, 2].astype(np.float32)
        m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        if not np.any(m):
            return h_map, valid
        uu = uu[m]
        vv = vv[m]
        zz = zz[m]
        h_map[vv, uu] = zz
        valid[vv, uu] = np.isfinite(zz) & (zz > float(z_eps))
        return h_map, valid
    except Exception:
        return h_map, valid


def _infer_hw_from_mesh(mesh: o3d.geometry.TriangleMesh) -> tuple[int, int]:
    """
    由 mesh 顶点的 (u,v) 推断当前高度场对应的图像尺寸 (H,W)。
    注意：这里假设 mesh 顶点覆盖规则像素网格（ImageProcessor 的 height-mesh 逻辑满足该假设）。
    """
    try:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        if v.size == 0:
            return (1, 1)
        u_max = float(np.nanmax(v[:, 0]))
        v_max = float(np.nanmax(v[:, 1]))
        W = int(max(1, int(np.rint(u_max)) + 1))
        H = int(max(1, int(np.rint(v_max)) + 1))
        return (H, W)
    except Exception:
        return (1, 1)


def _scale_corners_xy(
    corners: np.ndarray,
    *,
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
) -> np.ndarray:
    """
    将角点从 src_wh 坐标系缩放到 dst_wh：
      - corners: (N,1,2) 或 (N,2)
      - src_wh / dst_wh: (W,H)
    """
    c = np.asarray(corners, dtype=np.float64).reshape(-1, 2).copy()
    sw, sh = float(src_wh[0]), float(src_wh[1])
    dw, dh = float(dst_wh[0]), float(dst_wh[1])
    if sw <= 1 or sh <= 1:
        return c.astype(np.float32)
    sx = dw / sw
    sy = dh / sh
    c[:, 0] = c[:, 0] * sx
    c[:, 1] = c[:, 1] * sy
    return c.astype(np.float32)


def _scale_xy_points(
    pts_xy: np.ndarray,
    *,
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
) -> np.ndarray:
    """将任意 (N,2) 点从 src_wh 坐标系缩放到 dst_wh（W,H）。"""
    p = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2).copy()
    sw, sh = float(src_wh[0]), float(src_wh[1])
    dw, dh = float(dst_wh[0]), float(dst_wh[1])
    if sw <= 1 or sh <= 1:
        return p.astype(np.float32)
    p[:, 0] *= (dw / sw)
    p[:, 1] *= (dh / sh)
    return p.astype(np.float32)


def _match_features_sift_orb(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    *,
    method: str,
    max_features: int,
    ratio: float,
    max_matches: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    返回匹配点对：
      - prev_xy: (N,2)
      - cur_xy:  (N,2)
    method: 'sift' | 'orb' | 'auto'
    """
    m = (method or "auto").strip().lower()
    if m not in ("sift", "orb", "auto"):
        m = "auto"

    def _mk_detector(name: str):
        if name == "sift":
            try:
                det = cv2.SIFT_create(nfeatures=int(max(64, max_features)))
                return det, cv2.NORM_L2
            except Exception:
                return None, None
        # orb
        try:
            det = cv2.ORB_create(nfeatures=int(max(128, max_features)))
            return det, cv2.NORM_HAMMING
        except Exception:
            return None, None

    det, norm = _mk_detector(m if m != "auto" else "sift")
    used = m if m != "auto" else "sift"
    if det is None:
        det, norm = _mk_detector("orb")
        used = "orb"
    if det is None or norm is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), "none"

    # detect+compute
    try:
        k1, d1 = det.detectAndCompute(prev_gray, None)
        k2, d2 = det.detectAndCompute(cur_gray, None)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), used
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), used

    # knn match + ratio test
    try:
        bf = cv2.BFMatcher(normType=int(norm), crossCheck=False)
        knn = bf.knnMatch(d1, d2, k=2)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), used

    good = []
    rr = float(ratio)
    rr = 0.85 if (not np.isfinite(rr)) else float(np.clip(rr, 0.3, 0.99))
    for mm in knn:
        if mm is None or len(mm) < 2:
            continue
        a, b = mm[0], mm[1]
        if a.distance < rr * b.distance:
            good.append(a)
    if len(good) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), used

    # sort by distance and cap
    good = sorted(good, key=lambda x: float(x.distance))
    good = good[: int(max(8, max_matches))]

    prev_xy = np.array([k1[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 2)
    cur_xy = np.array([k2[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 2)
    return prev_xy, cur_xy, used


def _points_xy_to_uvh(
    pts_xy: np.ndarray,
    *,
    h_map: np.ndarray,
    h_valid: np.ndarray,
    step: int,
    fallback_to_bilinear: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """对 (N,2) 点采样 h，返回 (N,3) uvh 以及 ok_mask (N,)"""
    xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    n = int(xy.shape[0])
    uvh = np.full((n, 3), np.nan, dtype=np.float64)
    ok = np.zeros((n,), dtype=np.bool_)
    step_i = int(max(1, step))
    for i in range(n):
        u, v = float(xy[i, 0]), float(xy[i, 1])
        hh, okh = _sample_h_on_grid(
            h_map,
            h_valid,
            u=u,
            v=v,
            step=step_i,
            fallback_to_bilinear=bool(fallback_to_bilinear),
        )
        uvh[i, 0] = float(u)
        uvh[i, 1] = float(v)
        uvh[i, 2] = float(hh) if okh else float("nan")
        ok[i] = bool(okh)
    return uvh, ok


def _estimate_rigid_T_svd_ransac(
    cur_uvh: np.ndarray,
    prev_uvh: np.ndarray,
    prev_xy: np.ndarray,
    *,
    ok_mask: np.ndarray,
    iters: int,
    inlier_thresh_px: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    用 RANSAC 在 uvh 对应上估计刚体变换（SVD），以内点判据：||proj(T*cur_uvh) - prev_xy|| <= thresh。
    返回 (T, inlier_mask)。若失败，T=None。
    """
    m = np.asarray(ok_mask, dtype=np.bool_).reshape(-1)
    cu = np.asarray(cur_uvh, dtype=np.float64).reshape(-1, 3)
    pu = np.asarray(prev_uvh, dtype=np.float64).reshape(-1, 3)
    pxy = np.asarray(prev_xy, dtype=np.float64).reshape(-1, 2)
    if cu.shape[0] != pu.shape[0] or cu.shape[0] != pxy.shape[0] or cu.shape[0] != m.shape[0]:
        return None, np.zeros((0,), dtype=np.bool_)
    idx = np.where(m & np.isfinite(cu).all(axis=1) & np.isfinite(pu).all(axis=1) & np.isfinite(pxy).all(axis=1))[0]
    if idx.size < 6:
        return None, np.zeros((cu.shape[0],), dtype=np.bool_)

    thr = float(inlier_thresh_px)
    if (not np.isfinite(thr)) or thr <= 0:
        thr = 3.0
    thr = float(np.clip(thr, 0.5, 20.0))
    n_iter = int(max(10, iters))
    n_iter = int(min(5000, n_iter))

    best_inliers = None
    best_n = -1
    best_p95 = float("inf")
    rng = np.random.default_rng(int(os.environ.get("FEAT_RANSAC_SEED", "0") or "0"))

    for _ in range(n_iter):
        try:
            samp = rng.choice(idx, size=6, replace=False)
        except Exception:
            break
        Tcand = _estimate_rigid_T_svd(cu[samp], pu[samp])
        if Tcand is None:
            continue
        e = _corner_direct_errors_px(T_prev_cur=Tcand, cur_uvh=cu, prev_xy=pxy, ok_mask=m)
        if e.size == 0:
            continue
        inl = np.zeros((cu.shape[0],), dtype=np.bool_)
        # rebuild inliers on all ok points
        pred = _apply_T_to_uvh(Tcand, cu[m])
        d = pred[:, :2] - pxy[m]
        ee = np.sqrt(np.sum(d * d, axis=1))
        ok_e = np.isfinite(ee)
        inl_local = ok_e & (ee <= thr)
        # map back to full mask
        mm_idx = np.where(m)[0]
        inl[mm_idx[inl_local]] = True

        n_inl = int(np.count_nonzero(inl))
        if n_inl < 6:
            continue
        # robust tie-breaker: inlier p95
        e_inl = ee[inl_local]
        p95 = float(np.percentile(e_inl, 95)) if e_inl.size > 0 else float("inf")
        if (n_inl > best_n) or ((n_inl == best_n) and (p95 < best_p95)):
            best_n = n_inl
            best_p95 = p95
            best_inliers = inl

    if best_inliers is None or int(np.count_nonzero(best_inliers)) < 6:
        # fall back to plain SVD on all ok
        Tc = _estimate_rigid_T_svd(cu[m], pu[m])
        inl = (m.copy() if Tc is not None else np.zeros((cu.shape[0],), dtype=np.bool_))
        return Tc, inl

    # refine on inliers
    Tc = _estimate_rigid_T_svd(cu[best_inliers], pu[best_inliers])
    return Tc, best_inliers

def _corners_bbox_xy(corners_xy: np.ndarray) -> tuple[float, float, float, float] | None:
    """返回 corners 的 bbox: (min_x, min_y, max_x, max_y)。"""
    try:
        xy = np.asarray(corners_xy, dtype=np.float64).reshape(-1, 2)
        if xy.size == 0:
            return None
        mn = np.nanmin(xy, axis=0)
        mx = np.nanmax(xy, axis=0)
        if not np.isfinite(mn).all() or not np.isfinite(mx).all():
            return None
        return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])
    except Exception:
        return None


def _apply_T_to_uvh(T: np.ndarray, uvh: np.ndarray) -> np.ndarray:
    """对 (N,3) uvh 应用 4x4 变换，返回 (N,3)。"""
    P = np.asarray(uvh, dtype=np.float64).reshape(-1, 3)
    if P.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    ones = np.ones((P.shape[0], 1), dtype=np.float64)
    Ph = np.concatenate([P, ones], axis=1)
    Qh = (Ph @ np.asarray(T, dtype=np.float64).T)
    return Qh[:, :3].astype(np.float64)


def _corner_direct_errors_px(
    *,
    T_prev_cur: np.ndarray,
    cur_uvh: np.ndarray,
    prev_xy: np.ndarray,
    ok_mask: np.ndarray,
) -> np.ndarray:
    """
    角点一一对应误差（更严格，比最近邻更能反映“是否真正对齐”）：
      - 对 ok_mask 为 True 的角点，计算 ||proj(T*cur_uvh) - prev_xy||_2
    """
    try:
        m = np.asarray(ok_mask, dtype=np.bool_).reshape(-1)
        if m.size == 0 or int(np.count_nonzero(m)) == 0:
            return np.zeros((0,), dtype=np.float64)
        cur = np.asarray(cur_uvh, dtype=np.float64).reshape(-1, 3)
        prev = np.asarray(prev_xy, dtype=np.float64).reshape(-1, 2)
        if cur.shape[0] != prev.shape[0] or cur.shape[0] != m.shape[0]:
            return np.zeros((0,), dtype=np.float64)
        pred = _apply_T_to_uvh(T_prev_cur, cur[m])
        d = pred[:, :2] - prev[m]
        e = np.sqrt(np.sum(d * d, axis=1))
        e = e[np.isfinite(e)]
        return e.astype(np.float64)
    except Exception:
        return np.zeros((0,), dtype=np.float64)


def _nn_errors_px(pred_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    """对每个 pred 点，计算到 ref 点集合的最近邻距离（像素）。"""
    p = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(ref_xy, dtype=np.float64).reshape(-1, 2)
    if p.size == 0 or r.size == 0:
        return np.zeros((0,), dtype=np.float64)
    # (M,1,2) - (1,N,2) -> (M,N,2)
    d = p[:, None, :] - r[None, :, :]
    d2 = np.sum(d * d, axis=2)
    mn = np.sqrt(np.min(d2, axis=1))
    return mn.astype(np.float64)


def _nn_index_and_error_px(pred_xy: np.ndarray, ref_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 pred->ref 的最近邻索引与距离（像素）。"""
    p = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(ref_xy, dtype=np.float64).reshape(-1, 2)
    if p.size == 0 or r.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float64)
    d = p[:, None, :] - r[None, :, :]
    d2 = np.sum(d * d, axis=2)
    idx = np.argmin(d2, axis=1).astype(np.int32)
    err = np.sqrt(np.min(d2, axis=1)).astype(np.float64)
    return idx, err


def _draw_points(
    img_bgr: np.ndarray,
    pts_xy: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int = 3,
    thickness: int = -1,
) -> np.ndarray:
    """在图像上画点（pts_xy: Nx2 float/int）。"""
    out = img_bgr
    try:
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
        H, W = int(out.shape[0]), int(out.shape[1])
        for x, y in pts:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if 0 <= xi < W and 0 <= yi < H:
                cv2.circle(out, (xi, yi), int(radius), color, int(thickness), lineType=cv2.LINE_AA)
    except Exception:
        pass
    return out


def _draw_lines_pred_to_ref(
    img_bgr: np.ndarray,
    pred_xy: np.ndarray,
    ref_xy: np.ndarray,
    nn_idx: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 160, 40),
    max_lines: int = 64,
) -> np.ndarray:
    """画 pred->最近邻ref 的连线（用于直观看误差）。"""
    out = img_bgr
    try:
        p = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
        r = np.asarray(ref_xy, dtype=np.float64).reshape(-1, 2)
        idx = np.asarray(nn_idx, dtype=np.int32).reshape(-1)
        H, W = int(out.shape[0]), int(out.shape[1])
        n = int(min(p.shape[0], idx.shape[0], max_lines))
        for i in range(n):
            j = int(idx[i])
            if j < 0 or j >= int(r.shape[0]):
                continue
            x0, y0 = p[i]
            x1, y1 = r[j]
            x0i, y0i = int(round(float(x0))), int(round(float(y0)))
            x1i, y1i = int(round(float(x1))), int(round(float(y1)))
            if 0 <= x0i < W and 0 <= y0i < H and 0 <= x1i < W and 0 <= y1i < H:
                cv2.line(out, (x0i, y0i), (x1i, y1i), color, 1, lineType=cv2.LINE_AA)
    except Exception:
        pass
    return out


def _draw_lines_pairs(
    img_bgr: np.ndarray,
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 160, 40),
    max_lines: int = 128,
) -> np.ndarray:
    """画 a[i] -> b[i] 的连线（点是一一对应的），用于可视化误差。"""
    out = img_bgr
    try:
        a = np.asarray(a_xy, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(b_xy, dtype=np.float64).reshape(-1, 2)
        n = int(min(a.shape[0], b.shape[0], max_lines))
        if n <= 0:
            return out
        H, W = int(out.shape[0]), int(out.shape[1])
        for i in range(n):
            x0, y0 = a[i]
            x1, y1 = b[i]
            x0i, y0i = int(round(float(x0))), int(round(float(y0)))
            x1i, y1i = int(round(float(x1))), int(round(float(y1)))
            if 0 <= x0i < W and 0 <= y0i < H and 0 <= x1i < W and 0 <= y1i < H:
                cv2.line(out, (x0i, y0i), (x1i, y1i), color, 1, lineType=cv2.LINE_AA)
    except Exception:
        pass
    return out


def _quantize_uv(u: float, step: int) -> int:
    """把连续像素坐标量化到规则网格（step 的倍数），返回 int 像素坐标。"""
    s = int(max(1, int(step)))
    return int(round(float(u) / float(s)) * s)


def _sample_h_on_grid(
    h_map: np.ndarray,
    valid: np.ndarray,
    *,
    u: float,
    v: float,
    step: int,
    fallback_to_bilinear: bool,
) -> tuple[float, bool]:
    """
    在“稀疏网格(step>1)”情况下，从高度图采样 h：
      - 默认：量化到最近网格点再取值（更适合 step=4 这类网格）
      - 可选：如果量化点无效，再回退到 bilinear（要求四邻域 valid）
    """
    try:
        H, W = int(h_map.shape[0]), int(h_map.shape[1])
        uq = _quantize_uv(u, step)
        vq = _quantize_uv(v, step)
        if 0 <= uq < W and 0 <= vq < H and bool(valid[vq, uq]) and np.isfinite(h_map[vq, uq]):
            return float(h_map[vq, uq]), True
        if bool(fallback_to_bilinear):
            return _bilinear_sample_h(h_map, valid, float(u), float(v))
        return float("nan"), False
    except Exception:
        return float("nan"), False


def _bilinear_sample_h(h_map: np.ndarray, valid: np.ndarray, u: float, v: float) -> tuple[float, bool]:
    """双线性采样高度；要求四邻域均为 valid。"""
    try:
        H, W = int(h_map.shape[0]), int(h_map.shape[1])
        if not (np.isfinite(u) and np.isfinite(v)):
            return float("nan"), False
        if u < 0 or v < 0 or u > (W - 1) or v > (H - 1):
            return float("nan"), False
        x0 = int(np.floor(u))
        y0 = int(np.floor(v))
        x1 = min(W - 1, x0 + 1)
        y1 = min(H - 1, y0 + 1)
        if not (valid[y0, x0] and valid[y0, x1] and valid[y1, x0] and valid[y1, x1]):
            return float("nan"), False
        q00 = float(h_map[y0, x0])
        q10 = float(h_map[y0, x1])
        q01 = float(h_map[y1, x0])
        q11 = float(h_map[y1, x1])
        if not all(np.isfinite(q) for q in (q00, q10, q01, q11)):
            return float("nan"), False
        tx = float(u - x0)
        ty = float(v - y0)
        a = q00 * (1.0 - tx) + q10 * tx
        b = q01 * (1.0 - tx) + q11 * tx
        return float(a * (1.0 - ty) + b * ty), True
    except Exception:
        return float("nan"), False


def _dense_match_uv_from_T(
    *,
    T_prev_cur: np.ndarray,
    cur_h_map: np.ndarray,
    cur_valid: np.ndarray,
    prev_h_map: np.ndarray,
    prev_valid: np.ndarray,
    stride: int,
    tau_h: float,
    tau_fb: float,
    max_points: int,
    use_bilinear: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    利用伪3D对齐矩阵 T_prev<-cur，把“当前帧规则网格点 (u,v,h)”映射到上一帧，
    得到 2D 稠密匹配 (u_cur,v_cur)->(u_prev,v_prev)。

    过滤策略：
      - 上一帧 valid（mask内）约束
      - 高度一致性：|h_warp - H_prev(u_prev,v_prev)| < tau_h
      - forward-backward：inv(T) 回投影误差 < tau_fb（像素）
    """
    Hc, Wc = int(cur_h_map.shape[0]), int(cur_h_map.shape[1])
    Hp, Wp = int(prev_h_map.shape[0]), int(prev_h_map.shape[1])
    if Hc <= 0 or Wc <= 0 or Hp <= 0 or Wp <= 0:
        return {}, {"n_src": 0.0, "n_keep": 0.0, "keep_ratio": 0.0}

    T = np.asarray(T_prev_cur, dtype=np.float64)
    if T.shape != (4, 4) or (not np.isfinite(T).all()):
        return {}, {"n_src": 0.0, "n_keep": 0.0, "keep_ratio": 0.0}
    try:
        invT = np.linalg.inv(T)
    except Exception:
        invT = None

    stride_i = int(max(1, stride))
    ys, xs = np.where(cur_valid)
    if ys.size == 0:
        return {}, {"n_src": 0.0, "n_keep": 0.0, "keep_ratio": 0.0}
    if stride_i > 1:
        m = ((xs % stride_i) == 0) & ((ys % stride_i) == 0)
        xs = xs[m]
        ys = ys[m]
    n_src = int(xs.size)
    if n_src == 0:
        return {}, {"n_src": 0.0, "n_keep": 0.0, "keep_ratio": 0.0}
    if int(max_points) > 0 and n_src > int(max_points):
        rng = np.random.default_rng(0)
        pick = rng.choice(n_src, size=int(max_points), replace=False)
        xs = xs[pick]
        ys = ys[pick]
        n_src = int(xs.size)

    u_cur = xs.astype(np.float32)
    v_cur = ys.astype(np.float32)
    h_cur = cur_h_map[ys, xs].astype(np.float32)

    ones = np.ones((n_src,), dtype=np.float64)
    P = np.stack([u_cur.astype(np.float64), v_cur.astype(np.float64), h_cur.astype(np.float64), ones], axis=1)
    Pp = (P @ T.T)
    u_prev_f = Pp[:, 0].astype(np.float32)
    v_prev_f = Pp[:, 1].astype(np.float32)
    h_prev_warp = Pp[:, 2].astype(np.float32)

    inb = (u_prev_f >= 0) & (u_prev_f <= (Wp - 1)) & (v_prev_f >= 0) & (v_prev_f <= (Hp - 1))
    if not np.any(inb):
        return {}, {"n_src": float(n_src), "n_keep": 0.0, "keep_ratio": 0.0}

    u_prev_f = u_prev_f[inb]
    v_prev_f = v_prev_f[inb]
    h_prev_warp = h_prev_warp[inb]
    u_cur_k = u_cur[inb]
    v_cur_k = v_cur[inb]
    h_cur_k = h_cur[inb]

    n_mid = int(u_prev_f.size)
    h_prev_samp = np.full((n_mid,), np.nan, dtype=np.float32)
    ok_prev = np.zeros((n_mid,), dtype=np.bool_)

    if bool(use_bilinear):
        for i in range(n_mid):
            hp, ok = _bilinear_sample_h(prev_h_map, prev_valid, float(u_prev_f[i]), float(v_prev_f[i]))
            h_prev_samp[i] = float(hp)
            ok_prev[i] = bool(ok)
        up_i = np.rint(u_prev_f).astype(np.int32)
        vp_i = np.rint(v_prev_f).astype(np.int32)
    else:
        up_i = np.rint(u_prev_f).astype(np.int32)
        vp_i = np.rint(v_prev_f).astype(np.int32)
        ok = (up_i >= 0) & (up_i < Wp) & (vp_i >= 0) & (vp_i < Hp) & prev_valid[vp_i, up_i]
        h_prev_samp[ok] = prev_h_map[vp_i[ok], up_i[ok]].astype(np.float32)
        ok_prev = ok

    duv_round = np.sqrt((u_prev_f - up_i.astype(np.float32)) ** 2 + (v_prev_f - vp_i.astype(np.float32)) ** 2).astype(np.float32)
    dh = np.abs(h_prev_warp - h_prev_samp).astype(np.float32)
    h_ok = ok_prev & np.isfinite(h_prev_samp) & np.isfinite(h_prev_warp) & (dh <= float(tau_h))

    fb_err = np.full((n_mid,), np.nan, dtype=np.float32)
    if (invT is not None) and (float(tau_fb) > 0):
        ones2 = np.ones((n_mid,), dtype=np.float64)
        P2 = np.stack(
            [
                u_prev_f.astype(np.float64),
                v_prev_f.astype(np.float64),
                h_prev_samp.astype(np.float64),
                ones2,
            ],
            axis=1,
        )
        Pb = (P2 @ invT.T)
        ub = Pb[:, 0].astype(np.float32)
        vb = Pb[:, 1].astype(np.float32)
        fb_err = np.sqrt((ub - u_cur_k) ** 2 + (vb - v_cur_k) ** 2).astype(np.float32)
        fb_ok = np.isfinite(fb_err) & (fb_err <= float(tau_fb))
        keep = h_ok & fb_ok
    else:
        keep = h_ok

    if not np.any(keep):
        return {}, {"n_src": float(n_src), "n_keep": 0.0, "keep_ratio": 0.0}

    uv_cur = np.stack([u_cur_k[keep].astype(np.int32), v_cur_k[keep].astype(np.int32)], axis=1)
    uv_prev = np.stack([up_i[keep].astype(np.int32), vp_i[keep].astype(np.int32)], axis=1)

    out = {
        "uv_cur": uv_cur.astype(np.int32),
        "uv_prev_f": np.stack([u_prev_f[keep], v_prev_f[keep]], axis=1).astype(np.float32),
        "uv_prev": uv_prev.astype(np.int32),
        "h_cur": h_cur_k[keep].astype(np.float32),
        "h_prev": h_prev_samp[keep].astype(np.float32),
        "h_prev_warp": h_prev_warp[keep].astype(np.float32),
        "dh": dh[keep].astype(np.float32),
        "duv_round": duv_round[keep].astype(np.float32),
        "fb_err": fb_err[keep].astype(np.float32),
    }
    conf = np.exp(-np.minimum(out["dh"], float(tau_h)) / max(1e-6, float(tau_h))).astype(np.float32)
    if float(tau_fb) > 0:
        conf = conf * np.exp(-np.minimum(out["fb_err"], float(tau_fb)) / max(1e-6, float(tau_fb))).astype(np.float32)
    out["conf"] = conf

    stats = {
        "n_src": float(n_src),
        "n_keep": float(int(out["uv_cur"].shape[0])),
        "keep_ratio": float(int(out["uv_cur"].shape[0]) / max(1, n_src)),
        "mean_dh": float(np.nanmean(out["dh"])) if out["dh"].size > 0 else float("nan"),
        "mean_fb": float(np.nanmean(out["fb_err"])) if out["fb_err"].size > 0 else float("nan"),
    }
    return out, stats


def _get_hw_from_height_stats(height_stats: Any, fallback_hw: tuple[int, int]) -> tuple[int, int]:
    try:
        if isinstance(height_stats, dict):
            hh = int(height_stats.get("h", fallback_hw[0]))
            ww = int(height_stats.get("w", fallback_hw[1]))
            return max(1, hh), max(1, ww)
    except Exception:
        pass
    return max(1, int(fallback_hw[0])), max(1, int(fallback_hw[1]))


def _maybe_unnormalize_mesh_z(mesh: o3d.geometry.TriangleMesh, stats: dict[str, Any] | None) -> o3d.geometry.TriangleMesh:
    """
    可选：把高度场 mesh 的 z 从“归一化后的 z”反解回“输入 arr 的原始尺度”。
    这在使用 per-frame minmax 时，可以显著减少跨帧 z 尺度漂移对 ICP 的影响。
    """
    if not _as_bool(os.environ.get("MESH_Z_UNNORMALIZE", "0")):
        return mesh
    if not stats:
        return mesh
    try:
        s = float(stats.get("z_lin_scale", 0.0))
        b = float(stats.get("z_lin_offset", 0.0))
        if (not np.isfinite(s)) or abs(s) < 1e-12:
            return mesh
        v = np.asarray(mesh.vertices, dtype=np.float64)
        if v.size == 0:
            return mesh
        v2 = v.copy()
        # z = s * arr_raw + b  =>  arr_raw = (z - b) / s
        v2[:, 2] = (v2[:, 2] - b) / s
        m2 = o3d.geometry.TriangleMesh(mesh)
        m2.vertices = o3d.utility.Vector3dVector(v2)
        return m2
    except Exception:
        return mesh


def _mesh_to_pcd(
    mesh: o3d.geometry.TriangleMesh,
    *,
    sample_points: int,
) -> o3d.geometry.PointCloud:
    mesh = _ensure_mesh_normals(mesh)
    if int(sample_points) > 0:
        # Poisson disk 采样更均匀，但可能慢一点；uniform 更快
        use_poisson = _as_bool(os.environ.get("SAMPLE_POISSON", "1"), True)
        try:
            if use_poisson:
                pcd = mesh.sample_points_poisson_disk(number_of_points=int(sample_points), init_factor=5)
            else:
                pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        except Exception:
            pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_normals():
            pcd.normals = mesh.vertex_normals
    return pcd


def _save_pair_overlay_pcd(
    *,
    out_path: str,
    pcd_prev: o3d.geometry.PointCloud,
    pcd_cur_aligned: o3d.geometry.PointCloud,
) -> None:
    """
    保存“上一帧 vs 当前帧(对齐后)”的叠加点云，便于肉眼检查匹配质量：
      - prev: 绿色
      - cur_aligned: 红色
    """
    p0 = o3d.geometry.PointCloud(pcd_prev)
    p1 = o3d.geometry.PointCloud(pcd_cur_aligned)
    try:
        n0 = len(p0.points)
        n1 = len(p1.points)
        if n0 > 0:
            p0.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (n0, 1)))
        if n1 > 0:
            p1.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n1, 1)))
    except Exception:
        pass
    out = p0 + p1
    o3d.io.write_point_cloud(out_path, out, write_ascii=False, compressed=False)


def _make_camera_frustum_lineset(
    *,
    scale: float,
    color_rgb: tuple[float, float, float] = (1.0, 0.6, 0.1),
) -> o3d.geometry.LineSet:
    """
    生成一个“相机视锥”的 LineSet（在相机自身坐标系下）：
      - 原点 O
      - 近端平面四个角点（一个小金字塔）
    之后你可以对它 .transform(T_c2w) 放到世界坐标。
    """
    s = float(max(1e-6, scale))
    # 近端平面：z 取 +s，xy 取 +-0.6s（只是示意）
    z = 1.0 * s
    a = 0.6 * s
    pts = np.array(
        [
            [0.0, 0.0, 0.0],   # 0: origin
            [-a, -a, z],       # 1
            [ a, -a, z],       # 2
            [ a,  a, z],       # 3
            [-a,  a, z],       # 4
        ],
        dtype=np.float64,
    )
    lines = np.array(
        [
            [0, 1], [0, 2], [0, 3], [0, 4],  # rays
            [1, 2], [2, 3], [3, 4], [4, 1],  # near plane loop
        ],
        dtype=np.int32,
    )
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([color_rgb], dtype=np.float64), (lines.shape[0], 1)))
    return ls


def _pcd_preprocess(
    pcd: o3d.geometry.PointCloud,
    *,
    voxel_size: float,
    estimate_normals: bool,
) -> o3d.geometry.PointCloud:
    out = pcd
    if float(voxel_size) > 0:
        out = out.voxel_down_sample(voxel_size=float(voxel_size))
    if estimate_normals:
        # kNN 半径取 voxel 的倍数
        rad = float(max(1e-6, float(voxel_size) * 2.5)) if float(voxel_size) > 0 else 5.0
        out.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=60))
        try:
            out.orient_normals_consistent_tangent_plane(50)
        except Exception:
            pass
    return out


def _compute_fpfh(
    pcd: o3d.geometry.PointCloud,
    *,
    voxel_size: float,
) -> o3d.pipelines.registration.Feature:
    rad = float(max(1e-6, float(voxel_size) * 5.0))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=100),
    )


@dataclass
class RegResult:
    T: np.ndarray  # 4x4
    fitness: float
    rmse: float
    method: str


def _make_icp_estimation(icp_method: str):
    """
    构造 ICP estimation（可选鲁棒核）。
    需要 Open3D 版本支持：TransformationEstimationPointToPlane(loss=...)。
    """
    icp_method_l = icp_method.strip().lower()
    use_robust = _as_bool(os.environ.get("ICP_USE_ROBUST_KERNEL", "1"))
    if not use_robust:
        if icp_method_l == "point_to_plane":
            return o3d.pipelines.registration.TransformationEstimationPointToPlane(), True
        return o3d.pipelines.registration.TransformationEstimationPointToPoint(False), False

    loss_name = os.environ.get("ICP_ROBUST_KERNEL", "tukey").strip().lower()
    scale = float(os.environ.get("ICP_ROBUST_SCALE", "10.0"))
    loss = None
    try:
        if loss_name == "huber":
            loss = o3d.pipelines.registration.HuberLoss(scale)
        elif loss_name == "cauchy":
            loss = o3d.pipelines.registration.CauchyLoss(scale)
        else:
            loss = o3d.pipelines.registration.TukeyLoss(scale)
    except Exception:
        loss = None

    if icp_method_l == "point_to_plane":
        try:
            return o3d.pipelines.registration.TransformationEstimationPointToPlane(loss), True
        except TypeError:
            return o3d.pipelines.registration.TransformationEstimationPointToPlane(), True
    try:
        return o3d.pipelines.registration.TransformationEstimationPointToPoint(False, loss), False
    except TypeError:
        return o3d.pipelines.registration.TransformationEstimationPointToPoint(False), False


def _try_colored_icp(
    *,
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    max_corr: float,
    max_iter: int,
) -> RegResult | None:
    """
    Colored ICP refine（如果 Open3D 版本支持且点云带颜色）。
    对“伪3D高度场”这类数据，颜色（灰度/高度映射）有时能显著增强对齐稳定性。
    """
    if not _as_bool(os.environ.get("ICP_USE_COLORED", "1")):
        return None
    try:
        if (not src.has_colors()) or (not tgt.has_colors()):
            return None
        lam = float(os.environ.get("ICP_COLORED_LAMBDA_GEOM", "0.968"))
        est = o3d.pipelines.registration.TransformationEstimationForColoredICP(lam)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max(1, max_iter)))
        icp = o3d.pipelines.registration.registration_colored_icp(
            src,
            tgt,
            float(max(1e-6, max_corr)),
            np.asarray(init_T, dtype=np.float64),
            est,
            criteria,
        )
        T = np.asarray(icp.transformation, dtype=np.float64)
        fit = float(getattr(icp, "fitness", 0.0))
        rmse = float(getattr(icp, "inlier_rmse", float("inf")))
        if not np.isfinite(T).all():
            return None
        return RegResult(T=T, fitness=fit, rmse=rmse, method="colored_icp")
    except Exception:
        return None


def _register_pair_multiscale(
    *,
    pcd_src_raw: o3d.geometry.PointCloud,
    pcd_tgt_raw: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    icp_method: str,
    base_voxel: float,
    icp_max_corr: float,
    icp_max_iter: int,
    use_global_ransac: bool,
    ransac_voxel: float,
    ransac_max_corr_mult: float,
    ransac_max_iter: int,
    ransac_confidence: float,
) -> RegResult:
    """
    多尺度 ICP（coarse->fine）：
      - 每层对原始点云做 voxel_down_sample
      - 逐层 ICP refine（使用上一层结果作为 init）
      - 可选：最粗层先做一次 RANSAC 得到 init
    """
    T = np.asarray(init_T, dtype=np.float64)
    if T.shape != (4, 4):
        T = np.eye(4, dtype=np.float64)

    base = float(max(1e-6, base_voxel))
    pyr_env = os.environ.get("ICP_VOXEL_PYRAMID", "").strip()
    if pyr_env:
        voxels = _parse_float_list("ICP_VOXEL_PYRAMID", "")
    else:
        voxels = [base * 3.0, base * 1.5, base]
    voxels = [float(v) for v in voxels if float(v) > 0]
    if len(voxels) == 0:
        voxels = [base]

    iters_env = os.environ.get("ICP_LEVEL_ITERS", "").strip()
    if iters_env:
        iters = [int(max(1, round(v))) for v in _parse_float_list("ICP_LEVEL_ITERS", "")]
    else:
        iters = [max(10, int(icp_max_iter * 0.35)), max(15, int(icp_max_iter * 0.6)), max(20, int(icp_max_iter))]
    # pad/trim
    if len(iters) < len(voxels):
        iters = iters + [iters[-1]] * (len(voxels) - len(iters))
    iters = iters[: len(voxels)]

    est, need_normals = _make_icp_estimation(icp_method)

    max_corr_mult = float(os.environ.get("ICP_LEVEL_MAX_CORR_MULT", "3.0"))

    # optional global init on coarsest
    method_used = "msicp"
    if use_global_ransac:
        try:
            src0 = _pcd_preprocess(pcd_src_raw, voxel_size=float(ransac_voxel), estimate_normals=need_normals)
            tgt0 = _pcd_preprocess(pcd_tgt_raw, voxel_size=float(ransac_voxel), estimate_normals=need_normals)
            src_f = _compute_fpfh(src0, voxel_size=float(ransac_voxel))
            tgt_f = _compute_fpfh(tgt0, voxel_size=float(ransac_voxel))
            max_corr = float(max(1e-6, float(ransac_voxel) * float(ransac_max_corr_mult)))
            checker1 = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
            checker2 = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr)
            criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(int(ransac_max_iter), int(max(1, ransac_max_iter // 10)))
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src0, tgt0, src_f, tgt_f,
                mutual_filter=True,
                max_correspondence_distance=max_corr,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[checker1, checker2],
                criteria=criteria,
            )
            if ransac is not None and np.isfinite(np.asarray(ransac.transformation)).all():
                T = np.asarray(ransac.transformation, dtype=np.float64)
                method_used = "ransac+msicp"
        except Exception:
            pass

    last_fit = 0.0
    last_rmse = float("inf")
    for lvl, (vx, itn) in enumerate(zip(voxels, iters)):
        src = _pcd_preprocess(pcd_src_raw, voxel_size=float(vx), estimate_normals=need_normals)
        tgt = _pcd_preprocess(pcd_tgt_raw, voxel_size=float(vx), estimate_normals=need_normals)
        max_corr = float(max(1e-6, max(float(icp_max_corr), float(vx) * max_corr_mult)))
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max(1, itn)))
        icp = o3d.pipelines.registration.registration_icp(src, tgt, max_corr, T, est, criteria)
        T = np.asarray(icp.transformation, dtype=np.float64)
        last_fit = float(getattr(icp, "fitness", 0.0))
        last_rmse = float(getattr(icp, "inlier_rmse", float("inf")))
        if not np.isfinite(T).all():
            break

        # 可选：在细层额外做一次 Colored-ICP refine（更稳但更慢）
        if (lvl == len(voxels) - 1) and _as_bool(os.environ.get("ICP_USE_COLORED", "1")):
            rr = _try_colored_icp(src=src, tgt=tgt, init_T=T, max_corr=max_corr, max_iter=max(10, int(itn)))
            if rr is not None and np.isfinite(rr.fitness) and np.isfinite(rr.rmse):
                # 用 Colored-ICP 的结果替换（通常更对齐纹理/高度结构）
                T = rr.T
                last_fit = rr.fitness
                last_rmse = rr.rmse
                method_used = f"{method_used}+colored"
    return RegResult(T=T, fitness=last_fit, rmse=last_rmse, method=method_used)


def _register_pair(
    *,
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    icp_method: str,
    icp_max_corr: float,
    icp_max_iter: int,
    use_global_ransac: bool,
    ransac_voxel: float,
    ransac_max_corr_mult: float,
    ransac_max_iter: int,
    ransac_confidence: float,
) -> RegResult:
    """
    返回 T_{tgt <- src}，即把 src 变换到 tgt 坐标系。
    """
    method_used = "icp"
    T0 = np.asarray(init_T, dtype=np.float64)
    if T0.shape != (4, 4):
        T0 = np.eye(4, dtype=np.float64)

    # optional global init (overrides init_T)
    if use_global_ransac:
        method_used = "ransac+icp"
        # RANSAC 建议在更强下采样上做
        est_normals = icp_method.strip().lower() == "point_to_plane"
        src_ds = _pcd_preprocess(pcd_src, voxel_size=ransac_voxel, estimate_normals=est_normals)
        tgt_ds = _pcd_preprocess(pcd_tgt, voxel_size=ransac_voxel, estimate_normals=est_normals)
        src_f = _compute_fpfh(src_ds, voxel_size=ransac_voxel)
        tgt_f = _compute_fpfh(tgt_ds, voxel_size=ransac_voxel)

        max_corr = float(max(1e-6, float(ransac_voxel) * float(ransac_max_corr_mult)))
        checker1 = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
        checker2 = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr)
        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(int(ransac_max_iter), int(max(1, ransac_max_iter // 10)))
        try:
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_ds,
                tgt_ds,
                src_f,
                tgt_f,
                mutual_filter=True,
                max_correspondence_distance=max_corr,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[checker1, checker2],
                criteria=criteria,
            )
            if ransac is not None and np.isfinite(np.asarray(ransac.transformation)).all():
                T0 = np.asarray(ransac.transformation, dtype=np.float64)
        except Exception:
            pass

    # ICP refine
    est, need_normals = _make_icp_estimation(icp_method)
    if need_normals:
        if not pcd_src.has_normals():
            pcd_src.estimate_normals()
        if not pcd_tgt.has_normals():
            pcd_tgt.estimate_normals()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max(1, icp_max_iter)))
    icp = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_tgt,
        float(max(1e-6, icp_max_corr)),
        T0,
        est,
        criteria,
    )
    T = np.asarray(icp.transformation, dtype=np.float64)
    fit = float(getattr(icp, "fitness", 0.0))
    rmse = float(getattr(icp, "inlier_rmse", float("inf")))
    return RegResult(T=T, fitness=fit, rmse=rmse, method=method_used)


def _better_than(a: RegResult, b: RegResult) -> bool:
    """选择更好的配准结果：优先 fitness，其次 rmse。"""
    if not np.isfinite(a.fitness):
        return False
    if not np.isfinite(b.fitness):
        return True
    if a.fitness > b.fitness + 1e-9:
        return True
    if b.fitness > a.fitness + 1e-9:
        return False
    return a.rmse < b.rmse


def _need_recovery(fitness: float, rmse: float) -> bool:
    # 默认加严阈值，让恢复更早介入（环境变量可覆盖）
    fit_min = float(os.environ.get("RECOVERY_FITNESS_MIN", "0.995"))
    rmse_max = float(os.environ.get("RECOVERY_RMSE_MAX", "2.0"))
    if not np.isfinite(fitness) or not np.isfinite(rmse):
        return True
    return (fitness < fit_min) or (rmse > rmse_max)


def main() -> None:
    # line buffered output
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    input_dir = os.environ.get("INPUT_DIR", r"D:\reloc3r\Data_IMU_Camera_Pose_5\Lines_Photo _2")
    output_dir = os.environ.get("OUTPUT_DIR", r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line2_Ours_Camera_Pose_60f_eval_vis")
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # ============================================================
    # Resume support (minimal invasive):
    #   - progress.json: lightweight state (last frame, keyframe, z-ref stats, etc.)
    #   - progress_rt.npz: frames + pairwise/cumulative RTs for already processed frames
    # ============================================================
    resume_enable = _as_bool(os.environ.get("RESUME_ENABLE", "1"))
    resume_save_every = int(os.environ.get("RESUME_SAVE_EVERY", "1"))
    resume_save_every = max(1, resume_save_every)
    progress_json_path = os.path.join(output_dir, "progress.json")
    progress_rt_path = os.path.join(output_dir, "progress_rt.npz")

    # tee log
    log_path = os.path.join(output_dir, "console_output.log")
    try:
        log_f = open(log_path, "w", encoding="utf-8")
    except Exception:
        log_f = None

    class _Tee:
        def __init__(self, f, orig):
            self.f = f
            self.orig = orig

        def write(self, s):
            try:
                if self.f is not None:
                    self.f.write(s)
                    self.f.flush()
            except Exception:
                pass
            self.orig.write(s)
            self.orig.flush()

        def flush(self):
            try:
                if self.f is not None:
                    self.f.flush()
            except Exception:
                pass
            self.orig.flush()

    if log_f is not None:
        sys.stdout = _Tee(log_f, sys.__stdout__)
        sys.stderr = _Tee(log_f, sys.__stderr__)

    print("=" * 80)
    print(f"[MeshBetweenRT] start { _now() } pid={os.getpid()}")
    print(f"[MeshBetweenRT] input_dir={input_dir}")
    print(f"[MeshBetweenRT] output_dir={output_dir}")
    print("=" * 80)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"INPUT_DIR not found: {input_dir}")

    image_paths = _list_images(input_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"no images in: {input_dir}")

    max_frames = int(os.environ.get("MAX_FRAMES", "0"))
    frame_step = int(os.environ.get("FRAME_STEP", "1"))
    frame_step = max(1, frame_step)
    idxs = list(range(0, len(image_paths), frame_step))
    if max_frames > 0:
        idxs = idxs[:max_frames]
    print(f"[MeshBetweenRT] frames_total={len(image_paths)} frame_step={frame_step} max_frames={max_frames} -> frames_used={len(idxs)}", flush=True)

    # Decide resume start (we only resume if both progress files exist and match input_dir).
    start_k = 0
    resume_last_frame: int | None = None
    resume_key_frame_idx: int | None = None
    resume_T0_key = np.eye(4, dtype=np.float64)
    resume_ref_z_mu: float | None = None
    resume_ref_z_sigma: float | None = None
    if resume_enable and os.path.isfile(progress_json_path) and os.path.isfile(progress_rt_path):
        j = _load_json(progress_json_path)
        if isinstance(j, dict):
            j_in = str(j.get("input_dir", ""))
            if (not j_in) or (os.path.normpath(j_in) == os.path.normpath(str(input_dir))):
                resume_last_frame = j.get("last_frame", None)
                try:
                    resume_last_frame = int(resume_last_frame) if resume_last_frame is not None else None
                except Exception:
                    resume_last_frame = None
                try:
                    resume_key_frame_idx = int(j.get("key_frame_idx", -1))
                    if resume_key_frame_idx < 0:
                        resume_key_frame_idx = None
                except Exception:
                    resume_key_frame_idx = None
                try:
                    Tk = np.asarray(j.get("T0_key", np.eye(4)), dtype=np.float64).reshape(4, 4)
                    if np.isfinite(Tk).all():
                        resume_T0_key = Tk
                except Exception:
                    resume_T0_key = np.eye(4, dtype=np.float64)
                try:
                    mu = j.get("ref_z_mu", None)
                    sig = j.get("ref_z_sigma", None)
                    resume_ref_z_mu = float(mu) if mu is not None and np.isfinite(float(mu)) else None
                    resume_ref_z_sigma = float(sig) if sig is not None and np.isfinite(float(sig)) else None
                except Exception:
                    resume_ref_z_mu, resume_ref_z_sigma = None, None

                # load processed RTs and validate last frame
                frames_done, _, _ = _load_progress_rt_npz(progress_rt_path)
                if len(frames_done) > 0:
                    last_from_npz = int(frames_done[-1])
                    if resume_last_frame is None:
                        resume_last_frame = last_from_npz
                if (resume_last_frame is not None) and (resume_last_frame in idxs):
                    try:
                        start_k = int(idxs.index(int(resume_last_frame)) + 1)
                    except Exception:
                        start_k = 0

    if resume_enable and start_k > 0:
        print(f"[Resume] enabled. last_frame={resume_last_frame} -> start_k={start_k}/{len(idxs)}", flush=True)
    elif resume_enable:
        # fallback hint if files exist but resume not activated
        if os.path.isdir(vis_dir) and (not os.path.isfile(progress_rt_path)):
            last_vis = _infer_last_frame_from_vis(vis_dir)
            if last_vis is not None:
                print(f"[Resume][Hint] found vis outputs up to cur={last_vis}, but no progress_rt.npz. Rerun once with RESUME_ENABLE=1 to generate progress files.", flush=True)

    if start_k >= len(idxs):
        print("[Resume] already finished: no remaining frames.", flush=True)
        return

    # ============================================================
    # Optional: chessboard-based intrinsics/extrinsics (physical camera pose)
    # 注意：这套结果与本脚本的 uvh_heightfield 的“shape RT”不等价！
    # ============================================================
    chess_enable = _as_bool(os.environ.get("CHESSBOARD_ENABLE", "1"))
    # findChessboardCorners 的 size 参数是 inner-corners (cols x rows)：
    # - 新增棋盘格：9x6 个方格 => inner-corners 为 8x5
    chess_try_sizes = _parse_size_list("CHESSBOARD_TRY_SIZES", "9x8;8x7;8x5")
    chess_square = float(os.environ.get("CHESSBOARD_SQUARE_SIZE", "8.0"))  # default: 8mm (fallback)
    chess_square_map = _parse_square_size_map(
        "CHESSBOARD_SQUARE_SIZE_MAP",
        # 默认：新增 9x6 方格棋盘（inner=8x5，5mm）；其余尺寸沿用 CHESSBOARD_SQUARE_SIZE
        "8x5=5.0",
    )

    def _square_size_for(used_size: tuple[int, int] | None) -> float:
        if used_size is None:
            return float(chess_square)
        try:
            return float(chess_square_map.get(tuple(used_size), float(chess_square)))
        except Exception:
            return float(chess_square)
    chess_refine = _as_bool(os.environ.get("CHESSBOARD_REFINE", "1"))
    chess_intr_path = os.environ.get("CHESSBOARD_INTRINSICS_JSON", "").strip()
    chess_save_debug = _as_bool(os.environ.get("CHESSBOARD_SAVE_DEBUG", "0"))
    # eval: use chessboard corners to evaluate mesh->2D mapping
    chess_eval_mapping = _as_bool(os.environ.get("CHESSBOARD_EVAL_MAPPING", "1"))
    chess_eval_tau_px = float(os.environ.get("CHESSBOARD_EVAL_TAU_PX", "3.0"))
    chess_eval_fallback_bilinear = _as_bool(os.environ.get("CHESSBOARD_EVAL_FALLBACK_BILINEAR", "0"))
    chess_eval_save_vis = _as_bool(os.environ.get("CHESSBOARD_EVAL_SAVE_VIS", "1"))
    chess_eval_draw_lines = _as_bool(os.environ.get("CHESSBOARD_EVAL_DRAW_LINES", "1"))
    chess_eval_max_lines = int(os.environ.get("CHESSBOARD_EVAL_MAX_LINES", "56"))
    chess_export_corner_uvh = _as_bool(os.environ.get("CHESSBOARD_EXPORT_CORNER_UVH", "1"))
    chess_export_corner_uvh_ply = _as_bool(os.environ.get("CHESSBOARD_EXPORT_CORNER_UVH_PLY", "0"))
    # ROI / gating / z-stabilize knobs
    icp_roi_enable = _as_bool(os.environ.get("ICP_ROI_ENABLE", "1"))
    icp_roi_margin = float(os.environ.get("ICP_ROI_MARGIN", "20.0"))
    icp_roi_min_points = int(os.environ.get("ICP_ROI_MIN_POINTS", "800"))
    icp_roi_strict = _as_bool(os.environ.get("ICP_ROI_STRICT", "1"))
    roi_fail_action = os.environ.get("ROI_FAIL_ACTION", "corners_svd").strip().lower()
    if roi_fail_action not in ("corners_svd", "keyframe", "reject", "ransac"):
        roi_fail_action = "corners_svd"
    eval_gate_enable = _as_bool(os.environ.get("EVAL_GATE_ENABLE", "1"))
    eval_p95_max = float(os.environ.get("EVAL_P95_MAX", "5.0"))
    eval_ok_min = float(os.environ.get("EVAL_OK_RATIO_MIN", "0.6"))
    eval_reject_update = _as_bool(os.environ.get("EVAL_REJECT_UPDATE", "0"))
    use_corners_init = _as_bool(os.environ.get("USE_CORNERS_INIT", "1"))
    mesh_z_match_ref = _as_bool(os.environ.get("MESH_Z_MATCH_REF", "0"))
    # ============================================================
    # Optional: feature matching (SIFT/ORB) as a replacement for chessboard corners
    # ============================================================
    feat_enable = _as_bool(os.environ.get("FEAT_ENABLE", "0"))
    feat_method = os.environ.get("FEAT_METHOD", "auto").strip().lower()  # sift/orb/auto
    feat_max_features = int(os.environ.get("FEAT_MAX_FEATURES", "2000"))
    feat_max_matches = int(os.environ.get("FEAT_MAX_MATCHES", "400"))
    feat_min_matches = int(os.environ.get("FEAT_MIN_MATCHES", "40"))
    feat_ratio = float(os.environ.get("FEAT_RATIO", "0.75"))
    feat_init_override = _as_bool(os.environ.get("FEAT_INIT_OVERRIDE", "1"))
    feat_ransac_iters = int(os.environ.get("FEAT_RANSAC_ITERS", "250"))
    feat_ransac_thr_px = float(os.environ.get("FEAT_RANSAC_INLIER_THR_PX", "3.0"))
    feat_h_fallback_bilinear = _as_bool(os.environ.get("FEAT_H_FALLBACK_BILINEAR", "0"))
    feat_eval_save_vis = _as_bool(os.environ.get("FEAT_EVAL_SAVE_VIS", "1"))
    feat_eval_draw_lines = _as_bool(os.environ.get("FEAT_EVAL_DRAW_LINES", "1"))
    feat_eval_max_lines = int(os.environ.get("FEAT_EVAL_MAX_LINES", "160"))
    cb_corners: dict[int, np.ndarray] = {}
    cb_size_used: dict[int, tuple[int, int]] = {}
    cb_img_size: tuple[int, int] | None = None  # (w,h)
    cb_K: np.ndarray | None = None
    cb_dist: np.ndarray | None = None
    cb_T_c2w: dict[int, np.ndarray] = {}  # camera->board/world
    cb_T_w2c: dict[int, np.ndarray] = {}  # board/world->camera
    cb_reproj_rmse: dict[int, float] = {}
    cb_eval_rows: list[dict[str, Any]] = []
    feat_eval_rows: list[dict[str, Any]] = []
    # caches for processed corners and uvh per frame (same corner ordering)
    cb_corners_proc_xy: dict[int, np.ndarray] = {}
    cb_corners_uvh: dict[int, np.ndarray] = {}
    cb_corners_uvh_ok: dict[int, np.ndarray] = {}
    # reference z stats for optional cross-frame stabilization
    ref_z_mu: float | None = None
    ref_z_sigma: float | None = None

    if chess_enable:
        mtxt = ", ".join([f"{k[0]}x{k[1]}={v:g}" for k, v in chess_square_map.items()])
        print(
            f"[Chessboard] enable=1 try_sizes={chess_try_sizes} "
            f"square_size_default={chess_square} square_size_map={{{mtxt}}}",
            flush=True,
        )
        chess_calib_max_frames = int(os.environ.get("CHESSBOARD_CALIB_MAX_FRAMES", "80"))
        chess_calib_max_frames = max(0, chess_calib_max_frames)  # 0 means no cap
        chess_calib_min_frames = int(os.environ.get("CHESSBOARD_CALIB_MIN_FRAMES", "10"))
        chess_calib_min_frames = max(6, chess_calib_min_frames)
        chess_calib_skip = _as_bool(os.environ.get("CHESSBOARD_SKIP_CALIB_IF_NO_INTR", "0"))
        # (0) load intrinsics if provided
        if chess_intr_path:
            intr = _load_intrinsics_json(chess_intr_path)
            if intr is not None:
                cb_K, cb_dist = intr
                print(f"[Chessboard] intrinsics loaded: {chess_intr_path}", flush=True)
            else:
                print(f"[Chessboard][Warning] failed to load intrinsics: {chess_intr_path}", flush=True)

        # (1) pre-scan: detect corners for frames in idxs (for calibrateCamera / solvePnP)
        # 重要：角点检测/记录在 mesh 生成之前完成；后续只做“角点在高度场上的 h 采样”，不再重复检测。
        imgpoints: list[np.ndarray] = []
        objpoints: list[np.ndarray] = []
        used_frames: list[int] = []
        used_sizes: list[tuple[int, int]] = []

        for fi in idxs:
            try:
                p = image_paths[int(fi)]
            except Exception:
                continue
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if cb_img_size is None:
                cb_img_size = (int(im.shape[1]), int(im.shape[0]))
            ok, corners, used = _detect_chessboard_corners(im, try_sizes=chess_try_sizes, refine=chess_refine)
            if ok and corners is not None and used is not None:
                cb_corners[int(fi)] = corners
                cb_size_used[int(fi)] = used
                # only collect for calibration if intrinsics not loaded
                if cb_K is None or cb_dist is None:
                    objp = _make_chessboard_object_points(int(used[0]), int(used[1]), _square_size_for(used))
                    objpoints.append(objp.astype(np.float32))
                    imgpoints.append(corners.reshape(-1, 2).astype(np.float32))
                    used_frames.append(int(fi))
                    used_sizes.append(used)
                if chess_save_debug:
                    try:
                        dbg = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                        cv2.drawChessboardCorners(dbg, used, corners, True)
                        dbg_dir = os.path.join(output_dir, "chessboard_debug")
                        os.makedirs(dbg_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(dbg_dir, f"cb_{int(fi):06d}.png"), dbg)
                    except Exception:
                        pass

        n_det = int(len(cb_corners))
        print(f"[Chessboard] detected frames: {n_det}/{len(idxs)}", flush=True)

        # (2) calibrate intrinsics if needed
        if (cb_K is None or cb_dist is None):
            if chess_calib_skip:
                print("[Chessboard][Warning] intrinsics missing and CHESSBOARD_SKIP_CALIB_IF_NO_INTR=1 -> skip calibrateCamera.", flush=True)
            if cb_img_size is None:
                print("[Chessboard][Warning] no valid images for calibration.", flush=True)
            elif len(objpoints) < chess_calib_min_frames:
                print(f"[Chessboard][Warning] not enough detections for calibrateCamera: {len(objpoints)} (need ~{chess_calib_min_frames}+)", flush=True)
            elif not chess_calib_skip:
                # require consistent size for simplest pipeline; if mixed, still calibrate but warn
                if len(set(used_sizes)) > 1:
                    print(f"[Chessboard][Warning] mixed board sizes detected: {sorted(set(used_sizes))}. Calibration may be unreliable.", flush=True)
                try:
                    # optional cap: calibrateCamera with hundreds of frames can be very slow
                    n_all = int(len(objpoints))
                    if chess_calib_max_frames > 0 and n_all > chess_calib_max_frames:
                        # take roughly evenly spaced subset
                        idx_pick = np.linspace(0, n_all - 1, num=int(chess_calib_max_frames), dtype=np.int32)
                        objpoints2 = [objpoints[int(i)] for i in idx_pick]
                        imgpoints2 = [imgpoints[int(i)] for i in idx_pick]
                        used_frames2 = [used_frames[int(i)] for i in idx_pick]
                        print(f"[Chessboard] calibrateCamera: subsample {len(objpoints2)}/{n_all} frames (CHESSBOARD_CALIB_MAX_FRAMES={chess_calib_max_frames})", flush=True)
                    else:
                        objpoints2 = objpoints
                        imgpoints2 = imgpoints
                        used_frames2 = used_frames
                        print(f"[Chessboard] calibrateCamera: using {len(objpoints2)} frames", flush=True)

                    t_cal0 = time.time()
                    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                        objpoints2,
                        imgpoints2,
                        cb_img_size,
                        None,
                        None,
                    )
                    t_cal = (time.time() - t_cal0) * 1000.0
                    cb_K = np.asarray(K, dtype=np.float64)
                    cb_dist = np.asarray(dist, dtype=np.float64).reshape(-1)
                    out_intr = os.path.join(output_dir, "chessboard_intrinsics.json")
                    _save_intrinsics_json(
                        out_intr,
                        cb_K,
                        cb_dist,
                        extra={
                            "reproj_rms": float(ret),
                            "image_size_wh": [int(cb_img_size[0]), int(cb_img_size[1])],
                            "square_size_default": float(chess_square),
                            "square_size_map": {f"{k[0]}x{k[1]}": float(v) for k, v in chess_square_map.items()},
                            "try_sizes": [f"{s[0]}x{s[1]}" for s in chess_try_sizes],
                            "frames_used": used_frames2,
                            "calib_total_detections": int(len(used_frames)),
                            "calib_used_detections": int(len(used_frames2)),
                            "calib_ms": float(t_cal),
                        },
                    )
                    print(f"[Chessboard] intrinsics calibrated: reproj_rms={float(ret):.4f} ms={t_cal:.1f} saved={out_intr}", flush=True)
                except Exception as e:
                    print(f"[Chessboard][Warning] calibrateCamera failed: {e}", flush=True)

        # (3) solvePnP per frame (if intrinsics available)
        if (cb_K is not None) and (cb_dist is not None):
            for fi, corners in cb_corners.items():
                used = cb_size_used.get(int(fi), None)
                if used is None:
                    continue
                objp = _make_chessboard_object_points(int(used[0]), int(used[1]), _square_size_for(used))
                imgp = corners.reshape(-1, 2).astype(np.float32)
                try:
                    ok, rvec, tvec = cv2.solvePnP(objp, imgp, cb_K, cb_dist, flags=cv2.SOLVEPNP_ITERATIVE)
                    if not ok:
                        continue
                    R, _ = cv2.Rodrigues(rvec)
                    T_w2c = _rt_to_T(R, tvec)
                    T_c2w = np.linalg.inv(T_w2c)
                    cb_T_w2c[int(fi)] = T_w2c
                    cb_T_c2w[int(fi)] = T_c2w
                    # reprojection error for this frame
                    proj, _ = cv2.projectPoints(objp, rvec, tvec, cb_K, cb_dist)
                    proj = proj.reshape(-1, 2).astype(np.float64)
                    err = proj - imgp.astype(np.float64)
                    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
                    cb_reproj_rmse[int(fi)] = rmse
                except Exception:
                    continue
        else:
            print("[Chessboard][Warning] intrinsics unavailable -> skip solvePnP.", flush=True)

    sample_points = int(os.environ.get("SAMPLE_POINTS", "60000"))
    voxel_size = float(os.environ.get("VOXEL_SIZE", "2.0"))
    icp_method = os.environ.get("ICP_METHOD", "point_to_plane")
    icp_max_corr = float(os.environ.get("ICP_MAX_CORR", "10.0"))
    icp_max_iter = int(os.environ.get("ICP_MAX_ITER", "80"))
    icp_multiscale = _as_bool(os.environ.get("ICP_MULTISCALE", "1"))
    pose_graph_opt = _as_bool(os.environ.get("POSE_GRAPH_OPTIMIZE", "1"))
    pg_voxel = float(os.environ.get("POSE_GRAPH_VOXEL", "6.0"))
    pg_max_corr_mult = float(os.environ.get("POSE_GRAPH_MAX_CORR_MULT", "2.5"))

    use_global_ransac = _as_bool(os.environ.get("USE_GLOBAL_RANSAC", "0"))
    ransac_voxel = float(os.environ.get("RANSAC_VOXEL_SIZE", "5.0"))
    ransac_max_corr_mult = float(os.environ.get("RANSAC_MAX_CORR_MULT", "2.5"))
    ransac_max_iter = int(os.environ.get("RANSAC_MAX_ITER", "50000"))
    ransac_confidence = float(os.environ.get("RANSAC_CONFIDENCE", "0.999"))

    save_mesh_aligned = _as_bool(os.environ.get("SAVE_MESH_ALIGNED", "0"))
    save_pair_results = _as_bool(os.environ.get("SAVE_PAIR_RESULTS", "1"))
    print_rt = _as_bool(os.environ.get("PRINT_RT", "1"))
    print_rt_every = int(os.environ.get("PRINT_RT_EVERY", "1"))
    print_rt_every = max(1, print_rt_every)
    print_rt_mode = os.environ.get("PRINT_RT_MODE", "pairwise").strip().lower()
    if print_rt_mode not in ("pairwise", "cumulative", "both"):
        print_rt_mode = "pairwise"
    vis_open3d = _as_bool(os.environ.get("VIS_OPEN3D", "1"))
    vis_show_rt_frames = _as_bool(os.environ.get("VIS_SHOW_RT_FRAMES", "1"))
    vis_rt_every = int(os.environ.get("VIS_RT_EVERY", "1"))
    vis_rt_every = max(1, vis_rt_every)
    vis_frame_size_s = os.environ.get("VIS_FRAME_SIZE", "auto").strip().lower()
    vis_show_camera_frustum = _as_bool(os.environ.get("VIS_SHOW_CAMERA_FRUSTUM", "1"))
    vis_traj_use_all = _as_bool(os.environ.get("VIS_TRAJ_USE_ALL_FRAMES", "1"))
    pose_interp = os.environ.get("POSE_INTERPRETATION", "camera").strip().lower()
    if pose_interp not in ("camera", "object"):
        pose_interp = "camera"

    # coordinate consistency self-check
    print_mesh_stats = _as_bool(os.environ.get("PRINT_MESH_STATS", "1"))
    print_mesh_stats_every = int(os.environ.get("PRINT_MESH_STATS_EVERY", "1"))
    print_mesh_stats_every = max(1, print_mesh_stats_every)

    # create processor
    os.environ["USE_RENDERING_OPTIMIZATION"] = "0"
    processor = ImageProcessor()
    processor.output_dir = output_dir
    processor.o3d_step = int(os.environ.get("EXP_O3D_STEP", "1"))
    processor.mesh_cell_size = int(os.environ.get("EXP_MESH_CELL_SIZE", str(getattr(processor, "mesh_cell_size", 7))))
    processor.start_frame_idx = 0

    # results
    pairwise_T: dict[int, np.ndarray] = {}     # t -> T_{t-1<-t}
    cumulative_T: dict[int, np.ndarray] = {}   # t -> T_{0<-t}
    diag_rows: list[dict[str, Any]] = []
    # for pose-graph optimization (store small pcd only)
    pg_pcds: dict[int, o3d.geometry.PointCloud] = {}

    # state
    prev_mesh: o3d.geometry.TriangleMesh | None = None
    prev_pcd: o3d.geometry.PointCloud | None = None
    prev_pcd_raw: o3d.geometry.PointCloud | None = None
    prev_mesh_stats: dict[str, float] | None = None
    # dense 2D matching (use mesh vertices / height-map)
    prev_h_map: np.ndarray | None = None
    prev_h_valid: np.ndarray | None = None
    prev_hw: tuple[int, int] | None = None
    prev_img_gray: np.ndarray | None = None
    prev_img_wh: tuple[int, int] | None = None  # (W,H) in original image coord
    key_pcd_raw: o3d.geometry.PointCloud | None = None
    key_frame_idx: int | None = None
    T0_key = np.eye(4, dtype=np.float64)
    T0_prev = np.eye(4, dtype=np.float64)      # T_{0<-t-1}

    # For resume: load previous RTs if available
    progress_frames: list[int] = []
    progress_pairwise: list[np.ndarray] = []
    progress_cumulative: list[np.ndarray] = []
    if resume_enable and start_k > 0 and os.path.isfile(progress_rt_path):
        pf, pp, pc = _load_progress_rt_npz(progress_rt_path)
        if len(pf) > 0 and len(pf) == len(pp) == len(pc):
            progress_frames, progress_pairwise, progress_cumulative = pf, pp, pc
            for fidx, Tp, Tc in zip(progress_frames, progress_pairwise, progress_cumulative):
                pairwise_T[int(fidx)] = np.asarray(Tp, dtype=np.float64).reshape(4, 4)
                cumulative_T[int(fidx)] = np.asarray(Tc, dtype=np.float64).reshape(4, 4)
            # restore T0_prev from last frame
            try:
                last_f = int(progress_frames[-1])
                T0_prev = np.asarray(cumulative_T[last_f], dtype=np.float64).reshape(4, 4)
            except Exception:
                T0_prev = np.eye(4, dtype=np.float64)
            # restore keyframe info if available
            if resume_key_frame_idx is not None:
                key_frame_idx = int(resume_key_frame_idx)
                T0_key = np.asarray(resume_T0_key, dtype=np.float64).reshape(4, 4)
            # restore z-ref stats (for MESH_Z_MATCH_REF)
            if resume_ref_z_mu is not None and resume_ref_z_sigma is not None:
                ref_z_mu, ref_z_sigma = float(resume_ref_z_mu), float(resume_ref_z_sigma)

    # If not resumed, initialize frame0 pose.
    if len(cumulative_T) == 0 and len(idxs) > 0:
        cumulative_T[int(idxs[0])] = np.eye(4, dtype=np.float64)

    # Helper: save progress files (called after each processed frame)
    def _save_progress_checkpoint(*, last_frame: int) -> None:
        if not resume_enable:
            return
        try:
            j = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "frame_step": int(frame_step),
                "max_frames": int(max_frames),
                "last_frame": int(last_frame),
                "key_frame_idx": (int(key_frame_idx) if key_frame_idx is not None else None),
                "T0_key": np.asarray(T0_key, dtype=np.float64).reshape(4, 4).tolist(),
                "ref_z_mu": (float(ref_z_mu) if ref_z_mu is not None else None),
                "ref_z_sigma": (float(ref_z_sigma) if ref_z_sigma is not None else None),
                "note": "Resume checkpoint for Exp_Mehsh_Between_RT.py",
            }
            _save_json_atomic(progress_json_path, j)
            # write RT npz (frames are kept in ascending order)
            keys = sorted([int(k) for k in cumulative_T.keys()])
            pair_list = [np.asarray(pairwise_T.get(int(k), np.eye(4)), dtype=np.float64).reshape(4, 4) for k in keys]
            cum_list = [np.asarray(cumulative_T[int(k)], dtype=np.float64).reshape(4, 4) for k in keys]
            _save_progress_rt_npz(progress_rt_path, keys, pair_list, cum_list)
        except Exception:
            pass

    # If resuming, rebuild prev/keyframe point clouds from images, then continue from start_k.
    if resume_enable and start_k > 0 and resume_last_frame is not None:
        try:
            last_f = int(resume_last_frame)
            if last_f in idxs:
                img_path0 = image_paths[int(last_f)]
                img_bgr0 = cv2.imread(img_path0, cv2.IMREAD_COLOR)
                if img_bgr0 is not None:
                    # build last frame mesh/pcd as prev state
                    processor.process_image(img_bgr0, frame_idx=int(last_f), original_image_path=img_path0)
                    mesh0 = processor._mesh_cache[int(last_f)]
                    mesh0 = _ensure_mesh_normals(mesh0)
                    # unnormalize and z-match (if enabled) will be applied in the main loop, but for prev state it is ok to keep mesh0 as is
                    height_stats0 = None
                    try:
                        if hasattr(processor, "_height_mesh_stats_cache"):
                            height_stats0 = processor._height_mesh_stats_cache.get(int(last_f), None)
                    except Exception:
                        height_stats0 = None
                    mesh0 = _maybe_unnormalize_mesh_z(mesh0, height_stats0 if isinstance(height_stats0, dict) else None)
                    # compute chessboard uvh cache for prev frame (needed by ROI strict)
                    if chess_enable and (int(last_f) in cb_corners):
                        try:
                            Hm, Wm = _infer_hw_from_mesh(mesh0)
                            src_wh = (int(cb_img_size[0]), int(cb_img_size[1])) if cb_img_size is not None else (int(img_bgr0.shape[1]), int(img_bgr0.shape[0]))
                            dst_wh = (int(Wm), int(Hm))
                            corners_proc = _scale_corners_xy(cb_corners[int(last_f)], src_wh=src_wh, dst_wh=dst_wh)
                            z_eps_map = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
                            h_map, h_valid = _mesh_to_height_maps(mesh0, h=Hm, w=Wm, z_eps=z_eps_map)
                            step_i = int(getattr(processor, "o3d_step", int(os.environ.get("EXP_O3D_STEP", "1"))))
                            step_i = max(1, step_i)
                            uvh = np.full((int(corners_proc.shape[0]), 3), np.nan, dtype=np.float64)
                            ok = np.zeros((int(corners_proc.shape[0]),), dtype=np.bool_)
                            for cid, (u, v) in enumerate(corners_proc.astype(np.float32)):
                                hh, okh = _sample_h_on_grid(h_map, h_valid, u=float(u), v=float(v), step=step_i, fallback_to_bilinear=False)
                                uvh[int(cid), 0] = float(u)
                                uvh[int(cid), 1] = float(v)
                                uvh[int(cid), 2] = float(hh) if okh else float("nan")
                                ok[int(cid)] = bool(okh)
                            cb_corners_proc_xy[int(last_f)] = corners_proc.astype(np.float32)
                            cb_corners_uvh[int(last_f)] = uvh
                            cb_corners_uvh_ok[int(last_f)] = ok
                        except Exception:
                            pass

                    mesh_stats0 = _mesh_basic_stats(mesh0)
                    pcd_raw0 = _mesh_to_pcd(mesh0, sample_points=int(os.environ.get("SAMPLE_POINTS", "60000")))
                    z_eps0 = float(os.environ.get("FILTER_PCD_Z_EPS", "0.5"))
                    min_pts0 = int(os.environ.get("FILTER_PCD_MIN_POINTS", "500"))
                    pcd_raw0 = _filter_pcd_by_z(pcd_raw0, z_eps=z_eps0, min_points=min_pts0)
                    if chess_enable and icp_roi_enable and (int(last_f) in cb_corners_proc_xy):
                        bb0 = _corners_bbox_xy(cb_corners_proc_xy[int(last_f)])
                        if bb0 is not None:
                            x0, y0, x1, y1 = bb0
                            pcd_raw0 = _filter_pcd_by_xy_bbox(
                                pcd_raw0,
                                min_x=x0,
                                min_y=y0,
                                max_x=x1,
                                max_y=y1,
                                margin=float(icp_roi_margin),
                                min_points=int(icp_roi_min_points),
                            )
                    if _as_bool(os.environ.get("PCD_USE_HEIGHT_COLORS", "1")):
                        pcd_raw0 = _pcd_add_height_colors(pcd_raw0)
                    est_normals0 = icp_method.strip().lower() == "point_to_plane"
                    pcd0 = _pcd_preprocess(pcd_raw0, voxel_size=float(os.environ.get("VOXEL_SIZE", "2.0")), estimate_normals=est_normals0)

                    prev_mesh = mesh0
                    prev_pcd = pcd0
                    prev_pcd_raw = pcd_raw0
                    prev_mesh_stats = mesh_stats0
                    prev_img_gray = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2GRAY)
                    prev_img_wh = (int(img_bgr0.shape[1]), int(img_bgr0.shape[0]))

                    # rebuild keyframe raw pcd if needed
                    if key_frame_idx is not None and int(key_frame_idx) != int(last_f):
                        try:
                            kf = int(key_frame_idx)
                            imgk = cv2.imread(image_paths[int(kf)], cv2.IMREAD_COLOR)
                            if imgk is not None:
                                processor.process_image(imgk, frame_idx=int(kf), original_image_path=image_paths[int(kf)])
                                mk = _ensure_mesh_normals(processor._mesh_cache[int(kf)])
                                mk = _maybe_unnormalize_mesh_z(mk, None)
                                pk = _mesh_to_pcd(mk, sample_points=int(os.environ.get("SAMPLE_POINTS", "60000")))
                                pk = _filter_pcd_by_z(pk, z_eps=z_eps0, min_points=min_pts0)
                                if _as_bool(os.environ.get("PCD_USE_HEIGHT_COLORS", "1")):
                                    pk = _pcd_add_height_colors(pk)
                                key_pcd_raw = pk
                        except Exception:
                            pass
                    else:
                        key_pcd_raw = pcd_raw0
                        key_frame_idx = int(last_f)
                        T0_key = T0_prev.copy()
        except Exception as e:
            print(f"[Resume][Warning] failed to rebuild state: {e}. Fall back to full run.", flush=True)
            start_k = 0

    # Main processing loop (supports resume via start_k and prebuilt prev_* state)
    for k in range(start_k, len(idxs)):
        fi = int(idxs[k])
        img_path = image_paths[fi]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_gray = None
        try:
            if img_bgr is not None:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            img_gray = None
        if img_bgr is None:
            print(f"[Warning] cannot read: {img_path}, skip")
            continue

        # mesh build
        t0 = time.time()
        try:
            processor.process_image(img_bgr, frame_idx=int(fi), original_image_path=img_path)
            if (not hasattr(processor, "_mesh_cache")) or (int(fi) not in processor._mesh_cache):
                raise RuntimeError("mesh not cached")
            mesh = processor._mesh_cache[int(fi)]
            mesh = _ensure_mesh_normals(mesh)
        except Exception as e:
            print(f"[Warning] frame={fi} mesh build failed: {e}")
            continue
        t_mesh = (time.time() - t0) * 1000.0

        # pull per-frame height-mesh normalization stats (A_t) if available
        height_stats = None
        try:
            if hasattr(processor, "_height_mesh_stats_cache"):
                height_stats = processor._height_mesh_stats_cache.get(int(fi), None)
        except Exception:
            height_stats = None

        # optional: unnormalize z back to input-arr scale to reduce per-frame minmax drift
        mesh = _maybe_unnormalize_mesh_z(mesh, height_stats if isinstance(height_stats, dict) else None)

        # (C2) 可选：把每帧有效区域 z 线性匹配到参考帧分布，减小跨帧尺度漂移
        if mesh_z_match_ref:
            try:
                eps_z = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
                v = np.asarray(mesh.vertices, dtype=np.float64)
                if v.size > 0:
                    z = v[:, 2].astype(np.float64)
                    m = np.isfinite(z) & (z > eps_z)
                    if int(np.count_nonzero(m)) > 500:
                        mu = float(np.mean(z[m]))
                        sig = float(np.std(z[m]))
                        if (ref_z_mu is None) or (ref_z_sigma is None):
                            if np.isfinite(mu) and np.isfinite(sig) and sig > 1e-6:
                                ref_z_mu, ref_z_sigma = mu, sig
                        else:
                            if np.isfinite(mu) and np.isfinite(sig) and sig > 1e-6:
                                z2 = z.copy()
                                z2[m] = ((z2[m] - mu) / sig) * float(ref_z_sigma) + float(ref_z_mu)
                                v2 = v.copy()
                                v2[:, 2] = z2
                                mesh2 = o3d.geometry.TriangleMesh(mesh)
                                mesh2.vertices = o3d.utility.Vector3dVector(v2)
                                mesh = mesh2
            except Exception:
                pass

        # 预先为本帧角点建立“处理后坐标 + uvh”缓存（供 ROI / corners-init / eval）
        if chess_enable and (int(fi) in cb_corners):
            try:
                Hm, Wm = _infer_hw_from_mesh(mesh)
                if cb_img_size is None:
                    src_wh = (int(img_bgr.shape[1]), int(img_bgr.shape[0]))
                else:
                    src_wh = (int(cb_img_size[0]), int(cb_img_size[1]))
                dst_wh = (int(Wm), int(Hm))
                corners_proc = _scale_corners_xy(cb_corners[int(fi)], src_wh=src_wh, dst_wh=dst_wh)  # (N,2)
                z_eps_map = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
                h_map, h_valid = _mesh_to_height_maps(mesh, h=Hm, w=Wm, z_eps=z_eps_map)
                step_i = int(getattr(processor, "o3d_step", int(os.environ.get("EXP_O3D_STEP", "1"))))
                step_i = max(1, step_i)
                uvh = np.full((int(corners_proc.shape[0]), 3), np.nan, dtype=np.float64)
                ok = np.zeros((int(corners_proc.shape[0]),), dtype=np.bool_)
                for cid, (u, v) in enumerate(corners_proc.astype(np.float32)):
                    h, okh = _sample_h_on_grid(h_map, h_valid, u=float(u), v=float(v), step=step_i, fallback_to_bilinear=False)
                    uvh[int(cid), 0] = float(u)
                    uvh[int(cid), 1] = float(v)
                    uvh[int(cid), 2] = float(h) if okh else float("nan")
                    ok[int(cid)] = bool(okh)
                cb_corners_proc_xy[int(fi)] = corners_proc.astype(np.float32)
                cb_corners_uvh[int(fi)] = uvh
                cb_corners_uvh_ok[int(fi)] = ok
            except Exception:
                pass

        # ============================================================
        # (A) 导出“棋盘格角点对应的伪3D点 (u,v,h)”（可选）
        #     - 角点来自 pre-scan（原图坐标），这里按 mesh 尺寸做一次比例缩放到处理后坐标系
        #     - 再在高度图上采样 h，得到 (u,v,h)
        # ============================================================
        if chess_enable and chess_export_corner_uvh and (int(fi) in cb_corners):
            try:
                Hm, Wm = _infer_hw_from_mesh(mesh)
                if cb_img_size is None:
                    src_wh = (int(img_bgr.shape[1]), int(img_bgr.shape[0]))
                else:
                    src_wh = (int(cb_img_size[0]), int(cb_img_size[1]))
                dst_wh = (int(Wm), int(Hm))
                corners_proc = _scale_corners_xy(cb_corners[int(fi)], src_wh=src_wh, dst_wh=dst_wh)  # Nx2
                z_eps_map = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
                h_map, h_valid = _mesh_to_height_maps(mesh, h=Hm, w=Wm, z_eps=z_eps_map)
                step_i = int(getattr(processor, "o3d_step", int(os.environ.get("EXP_O3D_STEP", "1"))))
                step_i = max(1, step_i)
                rows_uvh: list[dict[str, Any]] = []
                pts_uvh: list[list[float]] = []
                for cid, (u, v) in enumerate(corners_proc.astype(np.float32)):
                    h, okh = _sample_h_on_grid(
                        h_map,
                        h_valid,
                        u=float(u),
                        v=float(v),
                        step=step_i,
                        fallback_to_bilinear=False,
                    )
                    uq = _quantize_uv(float(u), step_i)
                    vq = _quantize_uv(float(v), step_i)
                    rows_uvh.append({
                        "frame_idx": int(fi),
                        "corner_id": int(cid),
                        "u_raw": float(cb_corners[int(fi)].reshape(-1, 2)[cid, 0]),
                        "v_raw": float(cb_corners[int(fi)].reshape(-1, 2)[cid, 1]),
                        "u_proc": float(u),
                        "v_proc": float(v),
                        "u_grid": int(uq),
                        "v_grid": int(vq),
                        "h": float(h) if okh else float("nan"),
                        "h_ok": int(1 if okh else 0),
                        "src_w": int(src_wh[0]),
                        "src_h": int(src_wh[1]),
                        "dst_w": int(dst_wh[0]),
                        "dst_h": int(dst_wh[1]),
                        "step": int(step_i),
                    })
                    if okh:
                        pts_uvh.append([float(u), float(v), float(h)])

                out_dir = os.path.join(output_dir, "chessboard_corner_uvh")
                os.makedirs(out_dir, exist_ok=True)
                out_csv = os.path.join(out_dir, f"corner_uvh_{int(fi):06d}.csv")
                fcsv, out_csv_actual = _safe_open_for_write(out_csv, newline="\n")
                with fcsv:
                    fieldnames = list(rows_uvh[0].keys()) if rows_uvh else ["frame_idx", "corner_id"]
                    w = csv.DictWriter(fcsv, fieldnames=fieldnames)
                    w.writeheader()
                    for r in rows_uvh:
                        w.writerow(r)

                if chess_export_corner_uvh_ply and len(pts_uvh) > 0:
                    try:
                        pcd_cb = o3d.geometry.PointCloud()
                        pcd_cb.points = o3d.utility.Vector3dVector(np.asarray(pts_uvh, dtype=np.float64))
                        # cyan points
                        pcd_cb.colors = o3d.utility.Vector3dVector(
                            np.tile(np.array([[0.2, 1.0, 1.0]], dtype=np.float64), (len(pts_uvh), 1))
                        )
                        o3d.io.write_point_cloud(os.path.join(out_dir, f"corner_uvh_{int(fi):06d}.ply"), pcd_cb, write_ascii=False)
                    except Exception:
                        pass
            except Exception:
                pass

        # coordinate consistency self-check: bbox/center/diag/z + arr_min/max + z_map_mode
        mesh_stats = _mesh_basic_stats(mesh)
        if print_mesh_stats and (k % print_mesh_stats_every == 0):
            arr_min = height_stats.get("input_arr_min") if isinstance(height_stats, dict) else None
            arr_max = height_stats.get("input_arr_max") if isinstance(height_stats, dict) else None
            z_map_mode = height_stats.get("z_map_mode") if isinstance(height_stats, dict) else None
            z_scale = height_stats.get("z_scale") if isinstance(height_stats, dict) else None
            try:
                print(
                    f"[MeshStats][frame={fi}] diag={mesh_stats.get('diag',''):.3f} "
                    f"center=({mesh_stats.get('center_x',''):.3f},{mesh_stats.get('center_y',''):.3f},{mesh_stats.get('center_z',''):.3f}) "
                    f"z=[{mesh_stats.get('z_min',''):.3f},{mesh_stats.get('z_max',''):.3f}] mean={mesh_stats.get('z_mean',''):.3f} "
                    f"| nz_ratio={mesh_stats.get('nz_ratio',''):.4f} "
                    f"nz_center=({mesh_stats.get('nz_center_x', float('nan')):.3f},{mesh_stats.get('nz_center_y', float('nan')):.3f},{mesh_stats.get('nz_center_z', float('nan')):.3f}) "
                    f"nz_z_mean={mesh_stats.get('nz_z_mean', float('nan')):.3f} "
                    f"arr_min/max=({arr_min},{arr_max}) map={z_map_mode} z_scale={z_scale}",
                    flush=True,
                )
            except Exception:
                print(f"[MeshStats][frame={fi}] (failed to format stats)", flush=True)

        # pcd (keep raw for multiscale/keyframe)
        pcd_raw = _mesh_to_pcd(mesh, sample_points=sample_points)
        # 关键优化：过滤掉 z<=eps 的背景点（否则 ICP 很容易被“整张平面”拖偏）
        # 默认更强的背景抑制：过滤掉更接近0的“背景平面”，并要求更高的有效点数
        z_eps = float(os.environ.get("FILTER_PCD_Z_EPS", "0.5"))
        min_pts = int(os.environ.get("FILTER_PCD_MIN_POINTS", "500"))
        pcd_raw = _filter_pcd_by_z(pcd_raw, z_eps=z_eps, min_points=min_pts)

        # (B/D2) ROI：用棋盘格角点 bbox 裁剪点云（避免全图背景拖偏）
        if chess_enable and icp_roi_enable and (int(fi) in cb_corners_proc_xy):
            bb = _corners_bbox_xy(cb_corners_proc_xy[int(fi)])
            if bb is not None:
                x0, y0, x1, y1 = bb
                pcd_raw = _filter_pcd_by_xy_bbox(
                    pcd_raw,
                    min_x=x0,
                    min_y=y0,
                    max_x=x1,
                    max_y=y1,
                    margin=float(icp_roi_margin),
                    min_points=int(icp_roi_min_points),
                )

        # (C1/D1) 给点云加 height 颜色，增强 Colored-ICP 的稳定性（对光照更不敏感）
        if _as_bool(os.environ.get("PCD_USE_HEIGHT_COLORS", "1")):
            pcd_raw = _pcd_add_height_colors(pcd_raw)

        est_normals = icp_method.strip().lower() == "point_to_plane"
        pcd = _pcd_preprocess(pcd_raw, voxel_size=voxel_size, estimate_normals=est_normals)
        # store for pose-graph (stronger downsample to control memory)
        if pose_graph_opt:
            try:
                p_pg = _pcd_preprocess(pcd_raw, voxel_size=pg_voxel, estimate_normals=est_normals)
                pg_pcds[int(fi)] = p_pg
            except Exception:
                pass

        if k == 0 or prev_mesh is None or prev_pcd is None:
            # init dense-matching state (optional)
            if _as_bool(os.environ.get("DENSE_MATCH_ENABLE", "0")):
                try:
                    hh, ww = _get_hw_from_height_stats(height_stats, (int(img_bgr.shape[0]), int(img_bgr.shape[1])))
                    z_eps_map = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
                    prev_h_map, prev_h_valid = _mesh_to_height_maps(mesh, h=hh, w=ww, z_eps=z_eps_map)
                    prev_hw = (hh, ww)
                except Exception:
                    prev_h_map, prev_h_valid, prev_hw = None, None, None
            prev_mesh = mesh
            prev_pcd = pcd
            prev_pcd_raw = pcd_raw
            prev_mesh_stats = mesh_stats
            prev_img_gray = img_gray
            prev_img_wh = (int(img_bgr.shape[1]), int(img_bgr.shape[0])) if (img_bgr is not None) else None
            key_pcd_raw = pcd_raw
            key_frame_idx = int(fi)
            T0_key = np.eye(4, dtype=np.float64)
            T0_prev = np.eye(4, dtype=np.float64)
            pairwise_T[int(fi)] = np.eye(4, dtype=np.float64)
            cumulative_T[int(fi)] = np.eye(4, dtype=np.float64)
            diag_rows.append({
                "frame_idx": int(fi),
                "img": os.path.basename(img_path),
                "pair_fitness": 1.0,
                "pair_rmse": 0.0,
                "method": "identity",
                "ms_mesh": float(t_mesh),
            # chessboard pose diagnostics (optional)
            "cb_ok": int(1 if int(fi) in cb_T_c2w else 0),
            "cb_reproj_rmse_px": float(cb_reproj_rmse.get(int(fi), float("nan"))),
                "nz_center_dxy": "",
                "nz_ratio_mult": "",
                "nz_bbox_shift": "",
                "gate_jump": "",
                "gate_mode": "",
                **mesh_stats,
                "arr_min": (height_stats.get("input_arr_min") if isinstance(height_stats, dict) else ""),
                "arr_max": (height_stats.get("input_arr_max") if isinstance(height_stats, dict) else ""),
                "z_map_mode": (height_stats.get("z_map_mode") if isinstance(height_stats, dict) else ""),
                "z_scale": (height_stats.get("z_scale") if isinstance(height_stats, dict) else ""),
                "z_lin_scale": (height_stats.get("z_lin_scale") if isinstance(height_stats, dict) else ""),
                "z_lin_offset": (height_stats.get("z_lin_offset") if isinstance(height_stats, dict) else ""),
            })
            print(f"[Frame {fi}] init (mesh build {t_mesh:.1f}ms)")
            _save_progress_checkpoint(last_frame=int(fi))
            continue

        # register current to prev: T_{prev<-cur}
        init_T = np.eye(4, dtype=np.float64)
        # tracking init：用上一帧的 pairwise 近似作为初值（更稳一点）
        try:
            init_T = np.asarray(pairwise_T.get(int(idxs[k - 1]), np.eye(4)), dtype=np.float64)
        except Exception:
            init_T = np.eye(4, dtype=np.float64)

        # 额外初值：利用有效区域中心(nz_center)的平移先验（像素坐标系里非常有效）
        if _as_bool(os.environ.get("INIT_USE_NZ_CENTER_SHIFT", "1")) and prev_mesh_stats is not None:
            try:
                px = float(prev_mesh_stats.get("nz_center_x", float("nan")))
                py = float(prev_mesh_stats.get("nz_center_y", float("nan")))
                cx = float(mesh_stats.get("nz_center_x", float("nan")))
                cy = float(mesh_stats.get("nz_center_y", float("nan")))
                if np.isfinite(px) and np.isfinite(py) and np.isfinite(cx) and np.isfinite(cy):
                    dx = px - cx
                    dy = py - cy
                    dT = _make_delta_T(0.0, 0.0, 0.0, dx, dy, 0.0)
                    init_T = dT @ init_T
            except Exception:
                pass

        # (D3) corners-init：用角点对应的伪3D点(uvh)估计一个强初值（可选）
        corners_init_T = None
        if chess_enable and use_corners_init and (int(idxs[k - 1]) in cb_corners_uvh) and (int(fi) in cb_corners_uvh):
            try:
                prev_i = int(idxs[k - 1])
                cur_i = int(fi)
                uvh_prev = np.asarray(cb_corners_uvh.get(prev_i), dtype=np.float64)
                uvh_cur = np.asarray(cb_corners_uvh.get(cur_i), dtype=np.float64)
                okp = np.asarray(cb_corners_uvh_ok.get(prev_i), dtype=np.bool_)
                okc = np.asarray(cb_corners_uvh_ok.get(cur_i), dtype=np.bool_)
                ok = okp & okc & np.isfinite(uvh_prev).all(axis=1) & np.isfinite(uvh_cur).all(axis=1)
                if int(np.count_nonzero(ok)) >= 6:
                    Tc = _estimate_rigid_T_svd(uvh_cur[ok], uvh_prev[ok])
                    if Tc is not None:
                        corners_init_T = Tc
                        # 用角点初值覆盖/融合 tracking 初值（默认更信角点）
                        if _as_bool(os.environ.get("CORNERS_INIT_OVERRIDE", "1")):
                            init_T = np.asarray(corners_init_T, dtype=np.float64)
            except Exception:
                corners_init_T = None

        # ============================================================
        # (Feature-init) 无棋盘格时，用 SIFT/ORB 匹配点替代角点：
        #   - 匹配 (u,v) -> 采样 h -> 得到 (u,v,h)
        #   - RANSAC+SVD 得到 init_T / 以及评估用的 (prev_xy, cur_uvh, ok_mask)
        # ============================================================
        feat_prev_xy_proc = None
        feat_cur_uvh = None
        feat_ok_mask = None
        feat_init_T = None
        feat_used_method = ""
        feat_n_matches = 0
        feat_n_inliers = 0
        if feat_enable and (prev_img_gray is not None) and (img_gray is not None) and (prev_mesh is not None):
            # only activate when chessboard data not available for this pair
            has_cb_pair = bool(chess_enable and (int(idxs[k - 1]) in cb_corners) and (int(fi) in cb_corners))
            if not has_cb_pair:
                try:
                    prev_xy0, cur_xy0, used = _match_features_sift_orb(
                        prev_img_gray,
                        img_gray,
                        method=feat_method,
                        max_features=int(feat_max_features),
                        ratio=float(feat_ratio),
                        max_matches=int(feat_max_matches),
                    )
                    feat_used_method = used
                    feat_n_matches = int(prev_xy0.shape[0])
                    if feat_n_matches >= int(feat_min_matches):
                        # scale to processed coordinate system
                        Hp, Wp = _infer_hw_from_mesh(prev_mesh)
                        Hc, Wc = _infer_hw_from_mesh(mesh)
                        # original size as src (assume constant)
                        if prev_img_wh is None:
                            src_wh_prev = (int(prev_img_gray.shape[1]), int(prev_img_gray.shape[0]))
                        else:
                            src_wh_prev = (int(prev_img_wh[0]), int(prev_img_wh[1]))
                        src_wh_cur = (int(img_gray.shape[1]), int(img_gray.shape[0]))
                        prev_xy = _scale_xy_points(prev_xy0, src_wh=src_wh_prev, dst_wh=(Wp, Hp))
                        cur_xy = _scale_xy_points(cur_xy0, src_wh=src_wh_cur, dst_wh=(Wc, Hc))

                        # build height maps for sampling
                        z_eps_map = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
                        prev_h_map_e, prev_h_valid_e = _mesh_to_height_maps(prev_mesh, h=Hp, w=Wp, z_eps=z_eps_map)
                        cur_h_map_e, cur_h_valid_e = _mesh_to_height_maps(mesh, h=Hc, w=Wc, z_eps=z_eps_map)
                        step_i = int(getattr(processor, "o3d_step", int(os.environ.get("EXP_O3D_STEP", "1"))))
                        step_i = max(1, step_i)

                        prev_uvh, okp = _points_xy_to_uvh(
                            prev_xy,
                            h_map=prev_h_map_e,
                            h_valid=prev_h_valid_e,
                            step=step_i,
                            fallback_to_bilinear=bool(feat_h_fallback_bilinear),
                        )
                        cur_uvh, okc = _points_xy_to_uvh(
                            cur_xy,
                            h_map=cur_h_map_e,
                            h_valid=cur_h_valid_e,
                            step=step_i,
                            fallback_to_bilinear=bool(feat_h_fallback_bilinear),
                        )
                        ok = (okp & okc & np.isfinite(prev_uvh).all(axis=1) & np.isfinite(cur_uvh).all(axis=1))
                        if int(np.count_nonzero(ok)) >= int(feat_min_matches):
                            Tfeat, inl = _estimate_rigid_T_svd_ransac(
                                cur_uvh=cur_uvh,
                                prev_uvh=prev_uvh,
                                prev_xy=prev_xy,
                                ok_mask=ok,
                                iters=int(feat_ransac_iters),
                                inlier_thresh_px=float(feat_ransac_thr_px),
                            )
                            if Tfeat is not None:
                                feat_init_T = np.asarray(Tfeat, dtype=np.float64)
                                feat_prev_xy_proc = prev_xy
                                feat_cur_uvh = cur_uvh
                                feat_ok_mask = ok
                                feat_n_inliers = int(np.count_nonzero(inl)) if inl is not None else 0
                                if feat_init_override:
                                    init_T = feat_init_T.copy()
                except Exception:
                    pass

        # motion gate (for choosing recovery strategy)
        gate = _motion_gate(prev_mesh_stats, mesh_stats) if (prev_mesh_stats is not None) else {
            "nz_center_dxy": float("nan"),
            "nz_ratio_mult": float("nan"),
            "nz_bbox_shift": float("nan"),
            "gate_jump": 0,
            "gate_mode": "small",
        }

        # recovery flags/state (some branches below will override)
        recovered = False
        init_tag = "tracking"
        force_keyframe = False
        force_ransac = False
        skip_tracking_icp = False
        preset_disable_recovery = False
        # if ROI strict wants to pre-set a reg result (e.g., corners_svd / reject)
        preset_reg: RegResult | None = None
        preset_method: str | None = None

        # ============================================================
        # (B1/D2) ROI strict：如果 ROI 点数不足，绝不回退“全图点云”
        # 处理策略（ROI_FAIL_ACTION）：
        #   - corners_svd: 直接用角点SVD的刚体作为本帧对齐（不跑ICP）
        #   - keyframe: 强制走 keyframe recovery（需要后续 recovery block）
        #   - ransac: 强制开启 global ransac
        #   - reject: 直接拒绝更新（保持位姿不变）
        # ============================================================
        prev_i = int(idxs[k - 1])
        roi_failed = False
        pcd_src_raw_use = pcd_raw
        pcd_tgt_raw_use = prev_pcd_raw
        # (B1) chess corners missing is treated as ROI fail in strict mode (avoid full-image fallback)
        if chess_enable and icp_roi_enable and icp_roi_strict:
            if (prev_i not in cb_corners_proc_xy) or (int(fi) not in cb_corners_proc_xy) or (prev_pcd_raw is None):
                roi_failed = True

        if chess_enable and icp_roi_enable and (prev_i in cb_corners_proc_xy) and (int(fi) in cb_corners_proc_xy) and (prev_pcd_raw is not None):
            bb_cur = _corners_bbox_xy(cb_corners_proc_xy[int(fi)])
            bb_prev = _corners_bbox_xy(cb_corners_proc_xy[prev_i])
            if (bb_cur is not None) and (bb_prev is not None):
                x0, y0, x1, y1 = bb_cur
                a0, b0, a1, b1 = bb_prev
                src_roi = _filter_pcd_by_xy_bbox(
                    pcd_raw,
                    min_x=x0,
                    min_y=y0,
                    max_x=x1,
                    max_y=y1,
                    margin=float(icp_roi_margin),
                    min_points=int(icp_roi_min_points),
                )
                tgt_roi = _filter_pcd_by_xy_bbox(
                    prev_pcd_raw,
                    min_x=a0,
                    min_y=b0,
                    max_x=a1,
                    max_y=b1,
                    margin=float(icp_roi_margin),
                    min_points=int(icp_roi_min_points),
                )
                if (_pcd_len(src_roi) < int(icp_roi_min_points)) or (_pcd_len(tgt_roi) < int(icp_roi_min_points)):
                    roi_failed = True
                if (not roi_failed) or icp_roi_strict:
                    pcd_src_raw_use = src_roi
                    pcd_tgt_raw_use = tgt_roi

        # ROI fail action (skip tracking ICP if strict)
        if roi_failed and icp_roi_strict:
            if roi_fail_action == "keyframe":
                force_keyframe = True
                # still avoid full-image tracking ICP
                skip_tracking_icp = True
            elif roi_fail_action == "ransac":
                force_ransac = True
                # allow running (global) RANSAC-based registration as a recovery path
                skip_tracking_icp = False
            elif roi_fail_action == "reject":
                preset_reg = RegResult(T=np.eye(4, dtype=np.float64), fitness=0.0, rmse=float("inf"), method="roi_fail_reject")
                preset_method = "roi_fail:reject"
                recovered = True
                init_tag = "roi_fail:reject"
                skip_tracking_icp = True
                preset_disable_recovery = True  # hard reject: don't try other fallbacks
            else:
                # corners_svd default
                if corners_init_T is not None:
                    preset_reg = RegResult(T=np.asarray(corners_init_T, dtype=np.float64), fitness=1.0, rmse=0.0, method="corners_svd")
                    preset_method = "roi_fail:corners_svd"
                    recovered = True
                    init_tag = "roi_fail:corners_svd"
                    skip_tracking_icp = True
                else:
                    # no corners => fall back to keyframe (safer than full image)
                    force_keyframe = True
                    skip_tracking_icp = True

        # (D1) 动态 max_corr：如果上一帧对齐差，放宽对应距离
        icp_max_corr_local = float(icp_max_corr)
        if chess_enable and eval_gate_enable and (k >= 2):
            try:
                prev_eval = cb_eval_rows[-1] if len(cb_eval_rows) > 0 else None
                if isinstance(prev_eval, dict):
                    p95_prev = float(prev_eval.get("err_p95_px", float("nan")))
                    if np.isfinite(p95_prev) and p95_prev > float(eval_p95_max):
                        icp_max_corr_local = float(icp_max_corr) * 1.8
                        os.environ["ICP_ROBUST_SCALE"] = str(float(os.environ.get("ICP_ROBUST_SCALE", "10.0")) * 1.3)
            except Exception:
                pass

        # initial tracking registration (unless ROI strict pre-sets reg)
        t_reg = 0.0
        if preset_reg is not None:
            reg = preset_reg
        elif skip_tracking_icp:
            # should not happen often, but keep safe
            reg = RegResult(T=np.eye(4, dtype=np.float64), fitness=0.0, rmse=float("inf"), method="skip_tracking")
        else:
            t1 = time.time()
            use_global_ransac_local = bool(use_global_ransac or force_ransac)
            if icp_multiscale and (pcd_tgt_raw_use is not None):
                reg = _register_pair_multiscale(
                    pcd_src_raw=pcd_src_raw_use,
                    pcd_tgt_raw=pcd_tgt_raw_use,
                    init_T=init_T,
                    icp_method=icp_method,
                    base_voxel=voxel_size,
                    icp_max_corr=icp_max_corr_local,
                    icp_max_iter=icp_max_iter,
                    use_global_ransac=use_global_ransac_local,
                    ransac_voxel=ransac_voxel,
                    ransac_max_corr_mult=ransac_max_corr_mult,
                    ransac_max_iter=ransac_max_iter,
                    ransac_confidence=ransac_confidence,
                )
            else:
                reg = _register_pair(
                    pcd_src=pcd,
                    pcd_tgt=prev_pcd,
                    init_T=init_T,
                    icp_method=icp_method,
                    icp_max_corr=icp_max_corr_local,
                    icp_max_iter=icp_max_iter,
                    use_global_ransac=use_global_ransac_local,
                    ransac_voxel=ransac_voxel,
                    ransac_max_corr_mult=ransac_max_corr_mult,
                    ransac_max_iter=ransac_max_iter,
                    ransac_confidence=ransac_confidence,
                )
            t_reg = (time.time() - t1) * 1000.0

        # (D3 + 门控) 如果当前对齐很差，且有 corners-init，则用 corners-init 再跑一次并择优（按角点误差优先）
        if chess_enable and eval_gate_enable and (corners_init_T is not None) and (not preset_disable_recovery):
            try:
                prev_i = int(idxs[k - 1])
                cur_i = int(fi)
                if (prev_i in cb_corners_proc_xy) and (cur_i in cb_corners_uvh) and (prev_i in cb_corners_uvh_ok):
                    # evaluate current reg on direct correspondences
                    e1 = _corner_direct_errors_px(
                        T_prev_cur=reg.T,
                        cur_uvh=np.asarray(cb_corners_uvh[cur_i], dtype=np.float64),
                        prev_xy=np.asarray(cb_corners_proc_xy[prev_i], dtype=np.float64),
                        ok_mask=np.asarray(cb_corners_uvh_ok[prev_i] & cb_corners_uvh_ok[cur_i], dtype=np.bool_),
                    )
                    p95_1 = float(np.percentile(e1, 95)) if e1.size > 0 else float("inf")
                    ok_1 = float(np.mean(e1 <= float(chess_eval_tau_px))) if e1.size > 0 else 0.0
                    need_try = (p95_1 > float(eval_p95_max)) or (ok_1 < float(eval_ok_min))
                    if need_try:
                        # 强制恢复链：
                        # 1) corners_init -> ICP (优先ROI点云)
                        # 2) 放宽 corr / 增大迭代
                        # 3) 如果 ROI 打不开，再扩大 ROI margin（但严格模式下仍不回退全图）
                        rr_best: RegResult | None = None
                        p95_best = float("inf")
                        ok_best = 0.0

                        def _eval_reg(rrx: RegResult) -> tuple[float, float]:
                            ex = _corner_direct_errors_px(
                                T_prev_cur=rrx.T,
                                cur_uvh=np.asarray(cb_corners_uvh[cur_i], dtype=np.float64),
                                prev_xy=np.asarray(cb_corners_proc_xy[prev_i], dtype=np.float64),
                                ok_mask=np.asarray(cb_corners_uvh_ok[prev_i] & cb_corners_uvh_ok[cur_i], dtype=np.bool_),
                            )
                            p95x = float(np.percentile(ex, 95)) if ex.size > 0 else float("inf")
                            okx = float(np.mean(ex <= float(chess_eval_tau_px))) if ex.size > 0 else 0.0
                            return p95x, okx

                        def _try_icp(src_raw, tgt_raw, max_corr_x: float, max_iter_x: int) -> RegResult:
                            if icp_multiscale and (tgt_raw is not None):
                                return _register_pair_multiscale(
                                    pcd_src_raw=src_raw,
                                    pcd_tgt_raw=tgt_raw,
                                    init_T=np.asarray(corners_init_T, dtype=np.float64),
                                    icp_method=icp_method,
                                    base_voxel=voxel_size,
                                    icp_max_corr=max_corr_x,
                                    icp_max_iter=max_iter_x,
                                    use_global_ransac=False,
                                    ransac_voxel=ransac_voxel,
                                    ransac_max_corr_mult=ransac_max_corr_mult,
                                    ransac_max_iter=ransac_max_iter,
                                    ransac_confidence=ransac_confidence,
                                )
                            # single-scale fallback: build ds pcd from the provided raw (respect ROI)
                            est_normals = icp_method.strip().lower() == "point_to_plane"
                            src_ds = _pcd_preprocess(src_raw, voxel_size=float(voxel_size), estimate_normals=est_normals)
                            tgt_ds = _pcd_preprocess(tgt_raw, voxel_size=float(voxel_size), estimate_normals=est_normals)
                            return _register_pair(
                                pcd_src=src_ds,
                                pcd_tgt=tgt_ds,
                                init_T=np.asarray(corners_init_T, dtype=np.float64),
                                icp_method=icp_method,
                                icp_max_corr=max_corr_x,
                                icp_max_iter=max_iter_x,
                                use_global_ransac=False,
                                ransac_voxel=ransac_voxel,
                                ransac_max_corr_mult=ransac_max_corr_mult,
                                ransac_max_iter=ransac_max_iter,
                                ransac_confidence=ransac_confidence,
                            )

                        # (1) ROI ICP
                        try:
                            rr1 = _try_icp(pcd_src_raw_use, pcd_tgt_raw_use, float(icp_max_corr_local), int(icp_max_iter))
                            p95x, okx = _eval_reg(rr1)
                            rr_best, p95_best, ok_best = rr1, p95x, okx
                        except Exception:
                            pass

                        # (2) relax corr + iters
                        try:
                            rr2 = _try_icp(pcd_src_raw_use, pcd_tgt_raw_use, float(icp_max_corr_local) * 1.8, int(icp_max_iter * 1.5))
                            p95x, okx = _eval_reg(rr2)
                            if (p95x + 1e-6 < p95_best) or ((abs(p95x - p95_best) <= 1e-6) and (rr_best is not None) and _better_than(rr2, rr_best)):
                                rr_best, p95_best, ok_best = rr2, p95x, okx
                        except Exception:
                            pass

                        # (3) expand ROI margin (only if chess corners exist; strict mode still avoids full-image fallback)
                        if chess_enable and icp_roi_enable and (prev_i in cb_corners_proc_xy) and (cur_i in cb_corners_proc_xy) and (prev_pcd_raw is not None):
                            try:
                                bb_cur2 = _corners_bbox_xy(cb_corners_proc_xy[cur_i])
                                bb_prev2 = _corners_bbox_xy(cb_corners_proc_xy[prev_i])
                                if (bb_cur2 is not None) and (bb_prev2 is not None):
                                    x0, y0, x1, y1 = bb_cur2
                                    a0, b0, a1, b1 = bb_prev2
                                    mp2 = int(max(32, int(icp_roi_min_points * float(os.environ.get("ICP_ROI_RELAX_MIN_POINTS_MULT", "0.5")))))
                                    src2 = _filter_pcd_by_xy_bbox(pcd_raw, x0, y0, x1, y1, margin=float(icp_roi_margin) * 2.0, min_points=mp2)
                                    tgt2 = _filter_pcd_by_xy_bbox(prev_pcd_raw, a0, b0, a1, b1, margin=float(icp_roi_margin) * 2.0, min_points=mp2)
                                    if (_pcd_len(src2) >= mp2) and (_pcd_len(tgt2) >= mp2):
                                        rr3 = _try_icp(src2, tgt2, float(icp_max_corr_local) * 2.2, int(icp_max_iter * 1.8))
                                        p95x, okx = _eval_reg(rr3)
                                        if (p95x + 1e-6 < p95_best) or ((abs(p95x - p95_best) <= 1e-6) and (rr_best is not None) and _better_than(rr3, rr_best)):
                                            rr_best, p95_best, ok_best = rr3, p95x, okx
                            except Exception:
                                pass

                        rr = rr_best if rr_best is not None else reg
                        e2 = _corner_direct_errors_px(
                            T_prev_cur=rr.T,
                            cur_uvh=np.asarray(cb_corners_uvh[cur_i], dtype=np.float64),
                            prev_xy=np.asarray(cb_corners_proc_xy[prev_i], dtype=np.float64),
                            ok_mask=np.asarray(cb_corners_uvh_ok[prev_i] & cb_corners_uvh_ok[cur_i], dtype=np.bool_),
                        )
                        p95_2 = float(np.percentile(e2, 95)) if e2.size > 0 else float("inf")
                        ok_2 = float(np.mean(e2 <= float(chess_eval_tau_px))) if e2.size > 0 else 0.0
                        # choose by corner quality first, then fitness/rmse
                        if (p95_2 + 1e-6 < p95_1) or ((p95_2 <= p95_1 + 1e-6) and _better_than(rr, reg)):
                            reg = rr
                            init_tag = "corners_init"
                            recovered = True
            except Exception:
                pass

        # (Feature gate) 如果没有棋盘格角点但有特征点对应，也做同样的“坏了就 refine”的策略
        if feat_enable and eval_gate_enable and (feat_init_T is not None) and (feat_prev_xy_proc is not None) and (feat_cur_uvh is not None) and (feat_ok_mask is not None) and (not preset_disable_recovery):
            try:
                e1 = _corner_direct_errors_px(
                    T_prev_cur=reg.T,
                    cur_uvh=np.asarray(feat_cur_uvh, dtype=np.float64),
                    prev_xy=np.asarray(feat_prev_xy_proc, dtype=np.float64),
                    ok_mask=np.asarray(feat_ok_mask, dtype=np.bool_),
                )
                p95_1 = float(np.percentile(e1, 95)) if e1.size > 0 else float("inf")
                ok_1 = float(np.mean(e1 <= float(chess_eval_tau_px))) if e1.size > 0 else 0.0
                need_try = (p95_1 > float(eval_p95_max)) or (ok_1 < float(eval_ok_min))
                if need_try:
                    # 用 feat_init_T 做一次更强的 refine（放宽 corr / 增大 iter）
                    if icp_multiscale and (pcd_tgt_raw_use is not None):
                        rr = _register_pair_multiscale(
                            pcd_src_raw=pcd_src_raw_use,
                            pcd_tgt_raw=pcd_tgt_raw_use,
                            init_T=np.asarray(feat_init_T, dtype=np.float64),
                            icp_method=icp_method,
                            base_voxel=voxel_size,
                            icp_max_corr=float(icp_max_corr_local) * 1.8,
                            icp_max_iter=int(icp_max_iter * 1.5),
                            use_global_ransac=False,
                            ransac_voxel=ransac_voxel,
                            ransac_max_corr_mult=ransac_max_corr_mult,
                            ransac_max_iter=ransac_max_iter,
                            ransac_confidence=ransac_confidence,
                        )
                    else:
                        rr = _register_pair(
                            pcd_src=pcd,
                            pcd_tgt=prev_pcd,
                            init_T=np.asarray(feat_init_T, dtype=np.float64),
                            icp_method=icp_method,
                            icp_max_corr=float(icp_max_corr_local) * 1.8,
                            icp_max_iter=int(icp_max_iter * 1.5),
                            use_global_ransac=False,
                            ransac_voxel=ransac_voxel,
                            ransac_max_corr_mult=ransac_max_corr_mult,
                            ransac_max_iter=ransac_max_iter,
                            ransac_confidence=ransac_confidence,
                        )
                    e2 = _corner_direct_errors_px(
                        T_prev_cur=rr.T,
                        cur_uvh=np.asarray(feat_cur_uvh, dtype=np.float64),
                        prev_xy=np.asarray(feat_prev_xy_proc, dtype=np.float64),
                        ok_mask=np.asarray(feat_ok_mask, dtype=np.bool_),
                    )
                    p95_2 = float(np.percentile(e2, 95)) if e2.size > 0 else float("inf")
                    if (p95_2 + 1e-6 < p95_1) or ((p95_2 <= p95_1 + 1e-6) and _better_than(rr, reg)):
                        reg = rr
                        init_tag = f"feat_init:{feat_used_method}"
                        recovered = True

                # 记录/可视化（类似 chessboard_eval_vis）
                try:
                    n_total = int(np.asarray(feat_ok_mask, dtype=np.bool_).shape[0])
                    n_ok = int(np.count_nonzero(np.asarray(feat_ok_mask, dtype=np.bool_)))
                    feat_eval_rows.append({
                        "prev_frame": int(idxs[k - 1]),
                        "cur_frame": int(fi),
                        "method": str(feat_used_method),
                        "n_matches": int(feat_n_matches),
                        "n_ok_uvh": int(n_ok),
                        "n_inliers": int(feat_n_inliers),
                        "err_p95_px": float(p95_1),
                        "ok_ratio_tau": float(ok_1),
                        "tau_px": float(chess_eval_tau_px),
                        "eval_p95_max": float(eval_p95_max),
                        "eval_ok_min": float(eval_ok_min),
                        "note": "errors are computed on direct correspondences (feature matches) in processed uv-space",
                    })
                except Exception:
                    pass

                if feat_eval_save_vis:
                    try:
                        prev_i = int(idxs[k - 1])
                        cur_i = int(fi)
                        # build a prev-frame image in processed resolution (Wp,Hp)
                        prev_path = image_paths[int(prev_i)]
                        im_prev = cv2.imread(prev_path, cv2.IMREAD_COLOR)
                        if im_prev is not None and (prev_mesh is not None):
                            Hp, Wp = _infer_hw_from_mesh(prev_mesh)
                            im_prev_rs = cv2.resize(im_prev, (int(Wp), int(Hp)), interpolation=cv2.INTER_AREA)
                            m = np.asarray(feat_ok_mask, dtype=np.bool_).reshape(-1)
                            prev_xy_ok = np.asarray(feat_prev_xy_proc, dtype=np.float32).reshape(-1, 2)[m]
                            cur_uvh_ok = np.asarray(feat_cur_uvh, dtype=np.float64).reshape(-1, 3)[m]
                            pred = _apply_T_to_uvh(T_prev_cur, cur_uvh_ok).astype(np.float32)
                            pred_xy = pred[:, :2].astype(np.float32)

                            _draw_points(im_prev_rs, prev_xy_ok, color=(0, 255, 0), radius=2, thickness=-1)
                            if feat_eval_draw_lines:
                                _draw_lines_pairs(im_prev_rs, pred_xy, prev_xy_ok, max_lines=int(max(1, feat_eval_max_lines)))
                            _draw_points(im_prev_rs, pred_xy, color=(0, 0, 255), radius=2, thickness=-1)
                            txt = f"feat({feat_used_method}) prev={prev_i} cur={cur_i} p95={p95_1:.2f}px ok@{chess_eval_tau_px:g}px={ok_1:.2f}"
                            cv2.putText(im_prev_rs, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(im_prev_rs, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
                            vis_dir3 = os.path.join(output_dir, "feat_eval_vis")
                            os.makedirs(vis_dir3, exist_ok=True)
                            out_png = os.path.join(vis_dir3, f"eval_prev_{prev_i:06d}_cur_{cur_i:06d}.png")
                            cv2.imwrite(out_png, im_prev_rs)
                    except Exception:
                        pass
            except Exception:
                pass

        # recovery mode: motion-gated (small: greedy, medium: sphere sampling, jump: keyframe/backup)
        greedy_enabled = _as_bool(os.environ.get("USE_GREEDY_RECOVERY", "1"))
        greedy_print = _as_bool(os.environ.get("PRINT_GREEDY", "1"))
        greedy_icp_iter = int(os.environ.get("GREEDY_ICP_MAX_ITER", str(min(64, icp_max_iter))))
        greedy_icp_iter = max(5, greedy_icp_iter)
        ransac_recovery_user = _as_bool(os.environ.get("USE_RANSAC_RECOVERY", "1"))
        ransac_only_on_jump = _as_bool(os.environ.get("RANSAC_ONLY_ON_JUMP", "1"))
        gate_jump = int(gate.get("gate_jump", 0)) == 1
        gate_mode = str(gate.get("gate_mode", "small"))
        ransac_recovery = bool(ransac_recovery_user and ((not ransac_only_on_jump) or gate_jump))
        # recovered/init_tag may have been set by ROI strict pre-steps above

        use_keyframe_recovery = _as_bool(os.environ.get("USE_KEYFRAME_RECOVERY", "1"))
        keyframe_interval = int(os.environ.get("KEYFRAME_INTERVAL", "10"))
        keyframe_interval = max(1, keyframe_interval)

        # (A) if jump detected OR ROI failed, prefer keyframe recovery first
        ref_frame = int(idxs[k - 1])
        if (not preset_disable_recovery) and (gate_jump or force_keyframe) and _need_recovery(reg.fitness, reg.rmse) and use_keyframe_recovery and (key_pcd_raw is not None) and (key_frame_idx is not None):
            try:
                # init guess for key<-cur: inv(T0_key) @ (T0_prev @ init_T)
                T0_cur_guess = T0_prev @ np.asarray(init_T, dtype=np.float64)
                init_key = np.linalg.inv(T0_key) @ T0_cur_guess
                rk = _register_pair_multiscale(
                    pcd_src_raw=pcd_raw,
                    pcd_tgt_raw=key_pcd_raw,
                    init_T=init_key,
                    icp_method=icp_method,
                    base_voxel=voxel_size,
                    icp_max_corr=icp_max_corr,
                    icp_max_iter=icp_max_iter,
                    use_global_ransac=True,
                    ransac_voxel=ransac_voxel,
                    ransac_max_corr_mult=ransac_max_corr_mult,
                    ransac_max_iter=ransac_max_iter,
                    ransac_confidence=ransac_confidence,
                )
                if (not _need_recovery(rk.fitness, rk.rmse)) or _better_than(rk, reg):
                    # T0_cur = T0_key @ T_key<-cur
                    T0_cur_key = T0_key @ rk.T
                    # derive adjacent: T_prev<-cur = inv(T0_prev) @ T0_cur
                    T_prev_cur_key = np.linalg.inv(T0_prev) @ T0_cur_key
                    reg = RegResult(T=T_prev_cur_key, fitness=rk.fitness, rmse=rk.rmse, method=f"key:{rk.method}")
                    init_tag = f"keyframe:{key_frame_idx}"
                    recovered = True
                    ref_frame = int(key_frame_idx)
            except Exception:
                pass

        # (B) motion-gated local recovery (sphere sampling / greedy), only if still poor
        if (not preset_disable_recovery) and (not (roi_failed and icp_roi_enable and icp_roi_strict)) and _need_recovery(reg.fitness, reg.rmse) and greedy_enabled and (not gate_jump):
            t_rec0 = time.time()
            best = reg
            best_tag = "tracking"
            # choose candidate strategy
            use_sphere = (gate_mode == "medium") and _as_bool(os.environ.get("USE_SPHERE_SAMPLING", "1"))
            if use_sphere:
                cands = _sphere_pose_candidates(init_T=init_T, prev_stats=prev_mesh_stats, cur_stats=mesh_stats)
                cand_tag = "sphere"
            else:
                cands = _greedy_candidates(init_T)
                cand_tag = "greedy"
            if greedy_print:
                print(
                    f"[Recovery][frame={fi}] trigger (fit={reg.fitness:.3f}, rmse={reg.rmse:.4f}) "
                    f"gate={gate_mode} cands={len(cands)} mode={cand_tag} greedy_icp_iter={greedy_icp_iter}",
                    flush=True,
                )
            for tag, Tcand in cands:
                try:
                    rr = _register_pair(
                        pcd_src=pcd,
                        pcd_tgt=prev_pcd,
                        init_T=Tcand,
                        icp_method=icp_method,
                        icp_max_corr=icp_max_corr,
                        icp_max_iter=greedy_icp_iter,
                        use_global_ransac=False,
                        ransac_voxel=ransac_voxel,
                        ransac_max_corr_mult=ransac_max_corr_mult,
                        ransac_max_iter=ransac_max_iter,
                        ransac_confidence=ransac_confidence,
                    )
                    if _better_than(rr, best):
                        best = rr
                        best_tag = f"{cand_tag}:{tag}"
                except Exception:
                    continue

            # accept recovered if improved or at least meets thresholds
            if _better_than(best, reg) or (not _need_recovery(best.fitness, best.rmse)):
                if greedy_print:
                    dt = (time.time() - t_rec0) * 1000.0
                    print(
                        f"[Recovery][frame={fi}] choose={best_tag} "
                        f"(fit={best.fitness:.3f}, rmse={best.rmse:.4f}) cost={dt:.1f}ms",
                        flush=True,
                    )
                reg = best
                init_tag = best_tag
                recovered = True

        # (C) backup: if jump (or very poor) and still bad, allow RANSAC as a last resort
        if (not preset_disable_recovery) and (not (roi_failed and icp_roi_enable and icp_roi_strict)) and _need_recovery(reg.fitness, reg.rmse) and ransac_recovery:
            try:
                if icp_multiscale and (prev_pcd_raw is not None):
                    rr = _register_pair_multiscale(
                        pcd_src_raw=pcd_raw,
                        pcd_tgt_raw=prev_pcd_raw,
                        init_T=init_T,
                        icp_method=icp_method,
                        base_voxel=voxel_size,
                        icp_max_corr=icp_max_corr,
                        icp_max_iter=icp_max_iter,
                        use_global_ransac=True,
                        ransac_voxel=ransac_voxel,
                        ransac_max_corr_mult=ransac_max_corr_mult,
                        ransac_max_iter=ransac_max_iter,
                        ransac_confidence=ransac_confidence,
                    )
                else:
                    rr = _register_pair(
                        pcd_src=pcd,
                        pcd_tgt=prev_pcd,
                        init_T=init_T,
                        icp_method=icp_method,
                        icp_max_corr=icp_max_corr,
                        icp_max_iter=icp_max_iter,
                        use_global_ransac=True,
                        ransac_voxel=ransac_voxel,
                        ransac_max_corr_mult=ransac_max_corr_mult,
                        ransac_max_iter=ransac_max_iter,
                        ransac_confidence=ransac_confidence,
                    )
                if _better_than(rr, reg) or (not _need_recovery(rr.fitness, rr.rmse)):
                    reg = rr
                    init_tag = "ransac+icp"
                    recovered = True
            except Exception:
                pass

        T_prev_cur = reg.T
        # accumulate: T_{0<-cur} = T_{0<-prev} @ T_{prev<-cur}
        T0_cur = T0_prev @ T_prev_cur

        # (B1/D3) 评估门控：如果角点评估显示该帧对不可信，可选拒绝更新累计位姿
        if chess_enable and eval_gate_enable:
            try:
                prev_i = int(idxs[k - 1])
                cur_i = int(fi)
                if (prev_i in cb_corners_proc_xy) and (cur_i in cb_corners_uvh) and (prev_i in cb_corners_uvh_ok) and (cur_i in cb_corners_uvh_ok):
                    e = _corner_direct_errors_px(
                        T_prev_cur=T_prev_cur,
                        cur_uvh=np.asarray(cb_corners_uvh[cur_i], dtype=np.float64),
                        prev_xy=np.asarray(cb_corners_proc_xy[prev_i], dtype=np.float64),
                        ok_mask=np.asarray(cb_corners_uvh_ok[prev_i] & cb_corners_uvh_ok[cur_i], dtype=np.bool_),
                    )
                    p95_e = float(np.percentile(e, 95)) if e.size > 0 else float("inf")
                    ok_e = float(np.mean(e <= float(chess_eval_tau_px))) if e.size > 0 else 0.0
                    bad = (p95_e > float(eval_p95_max)) or (ok_e < float(eval_ok_min))
                    if bad and eval_reject_update:
                        # reject: keep last pose, mark as recovered/bad
                        T_prev_cur = np.eye(4, dtype=np.float64)
                        T0_cur = T0_prev.copy()
                        reg = RegResult(T=T_prev_cur, fitness=float(reg.fitness), rmse=float(reg.rmse), method=f"{reg.method}+rejected")
                        recovered = True
                        init_tag = f"{init_tag}+gate_reject"
            except Exception:
                pass

        # (Feature gate) 无棋盘格时，用特征点对应做门控（可选拒绝更新）
        if (not chess_enable) and feat_enable and eval_gate_enable and eval_reject_update:
            try:
                if (feat_prev_xy_proc is not None) and (feat_cur_uvh is not None) and (feat_ok_mask is not None):
                    e = _corner_direct_errors_px(
                        T_prev_cur=T_prev_cur,
                        cur_uvh=np.asarray(feat_cur_uvh, dtype=np.float64),
                        prev_xy=np.asarray(feat_prev_xy_proc, dtype=np.float64),
                        ok_mask=np.asarray(feat_ok_mask, dtype=np.bool_),
                    )
                    p95_e = float(np.percentile(e, 95)) if e.size > 0 else float("inf")
                    ok_e = float(np.mean(e <= float(chess_eval_tau_px))) if e.size > 0 else 0.0
                    bad = (p95_e > float(eval_p95_max)) or (ok_e < float(eval_ok_min))
                    if bad:
                        T_prev_cur = np.eye(4, dtype=np.float64)
                        T0_cur = T0_prev.copy()
                        reg = RegResult(T=T_prev_cur, fitness=float(reg.fitness), rmse=float(reg.rmse), method=f"{reg.method}+feat_rejected")
                        recovered = True
                        init_tag = f"{init_tag}+feat_gate_reject"
            except Exception:
                pass

        pairwise_T[int(fi)] = T_prev_cur
        cumulative_T[int(fi)] = T0_cur

        # ============================================================
        # (Eval) 用棋盘格角点验证“mesh->2D 映射”（像素层面）
        # 说明：mesh/RT 的 (u,v) 属于 ImageProcessor 处理后的坐标系，角点默认来自原图；
        #       这里用“按分辨率缩放”的近似把角点映射到处理后坐标系再评估。
        # ============================================================
        if chess_enable and chess_eval_mapping:
            try:
                prev_i = int(idxs[k - 1])
                # need corners in both frames
                if (prev_i in cb_corners) and (int(fi) in cb_corners) and (prev_mesh is not None):
                    # infer processed sizes from meshes
                    Hp, Wp = _infer_hw_from_mesh(prev_mesh)
                    Hc, Wc = _infer_hw_from_mesh(mesh)
                    # use original image size (from chess pre-scan) as src size
                    if cb_img_size is None:
                        src_wh = (int(img_bgr.shape[1]), int(img_bgr.shape[0]))
                    else:
                        src_wh = (int(cb_img_size[0]), int(cb_img_size[1]))
                    # scale corners into processed (u,v) coordinate system
                    prev_xy = _scale_corners_xy(cb_corners[prev_i], src_wh=src_wh, dst_wh=(Wp, Hp))
                    cur_xy = _scale_corners_xy(cb_corners[int(fi)], src_wh=src_wh, dst_wh=(Wc, Hc))

                    # build height maps (only for sampling h at chessboard corners)
                    z_eps_map = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
                    prev_h_map_e, prev_h_valid_e = _mesh_to_height_maps(prev_mesh, h=Hp, w=Wp, z_eps=z_eps_map)
                    cur_h_map_e, cur_h_valid_e = _mesh_to_height_maps(mesh, h=Hc, w=Wc, z_eps=z_eps_map)

                    # for each current corner: sample h then transform by T_prev<-cur
                    pred_prev: list[list[float]] = []
                    n_total = int(cur_xy.shape[0])
                    n_h_ok = 0
                    # mesh vertices are on a regular grid with stride == processor.o3d_step
                    step_i = int(getattr(processor, "o3d_step", int(os.environ.get("EXP_O3D_STEP", "1"))))
                    step_i = max(1, step_i)
                    for (u, v) in cur_xy.astype(np.float32):
                        hc, okh = _sample_h_on_grid(
                            cur_h_map_e,
                            cur_h_valid_e,
                            u=float(u),
                            v=float(v),
                            step=step_i,
                            fallback_to_bilinear=chess_eval_fallback_bilinear,
                        )
                        if not okh:
                            continue
                        n_h_ok += 1
                        P = np.array([float(u), float(v), float(hc), 1.0], dtype=np.float64)
                        Pp = (np.asarray(T_prev_cur, dtype=np.float64) @ P.reshape(4, 1)).reshape(4)
                        pred_prev.append([float(Pp[0]), float(Pp[1])])
                    pred_prev_xy = np.asarray(pred_prev, dtype=np.float32).reshape(-1, 2)

                    nn_idx, errs = _nn_index_and_error_px(pred_prev_xy, prev_xy)
                    if errs.size > 0:
                        mean_e = float(np.mean(errs))
                        med_e = float(np.median(errs))
                        p95_e = float(np.percentile(errs, 95))
                        ok_ratio = float(np.mean(errs <= float(chess_eval_tau_px)))
                    else:
                        mean_e = float("nan")
                        med_e = float("nan")
                        p95_e = float("nan")
                        ok_ratio = 0.0

                    cb_eval_rows.append({
                        "prev_frame": int(prev_i),
                        "cur_frame": int(fi),
                        "n_prev_corners": int(prev_xy.shape[0]),
                        "n_cur_corners": int(cur_xy.shape[0]),
                        "n_cur_h_ok": int(n_h_ok),
                        "n_pred": int(pred_prev_xy.shape[0]),
                        "err_mean_px": mean_e,
                        "err_median_px": med_e,
                        "err_p95_px": p95_e,
                        "ok_ratio_tau": ok_ratio,
                        "tau_px": float(chess_eval_tau_px),
                    })

                    # save visualization image (optional)
                    if chess_eval_save_vis:
                        try:
                            vis_dir2 = os.path.join(output_dir, "chessboard_eval_vis")
                            os.makedirs(vis_dir2, exist_ok=True)
                            # build a prev-frame image in processed resolution (Wp,Hp)
                            prev_path = image_paths[int(prev_i)]
                            im_prev = cv2.imread(prev_path, cv2.IMREAD_COLOR)
                            if im_prev is not None:
                                im_prev_rs = cv2.resize(im_prev, (int(Wp), int(Hp)), interpolation=cv2.INTER_AREA)
                                # draw: prev corners (green) + predicted corners (red)
                                _draw_points(im_prev_rs, prev_xy, color=(0, 255, 0), radius=3, thickness=-1)
                                if chess_eval_draw_lines and pred_prev_xy.size > 0 and nn_idx.size > 0:
                                    _draw_lines_pred_to_ref(im_prev_rs, pred_prev_xy, prev_xy, nn_idx, max_lines=int(max(1, chess_eval_max_lines)))
                                _draw_points(im_prev_rs, pred_prev_xy, color=(0, 0, 255), radius=3, thickness=-1)
                                # put stats
                                txt = f"prev={prev_i} cur={int(fi)} mean={mean_e:.2f}px med={med_e:.2f}px p95={p95_e:.2f}px ok@{chess_eval_tau_px:g}px={ok_ratio:.2f}"
                                cv2.putText(im_prev_rs, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                cv2.putText(im_prev_rs, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
                                out_png = os.path.join(vis_dir2, f"eval_prev_{prev_i:06d}_cur_{int(fi):06d}.png")
                                cv2.imwrite(out_png, im_prev_rs)
                        except Exception:
                            pass
            except Exception:
                pass

        # keep previous frame image for feature matching
        prev_img_gray = img_gray
        prev_img_wh = (int(img_bgr.shape[1]), int(img_bgr.shape[0])) if (img_bgr is not None) else prev_img_wh

        # ============================================================
        # (1-5) 稠密 2D 点-点匹配（由伪3D对齐 T 回投影得到）
        # ============================================================
        dense_stats: dict[str, float] = {}
        if _as_bool(os.environ.get("DENSE_MATCH_ENABLE", "0")) and (prev_h_map is not None) and (prev_h_valid is not None) and (prev_hw is not None):
            try:
                hh, ww = _get_hw_from_height_stats(height_stats, prev_hw)
                z_eps_map = float(os.environ.get("MESH_STATS_Z_EPS", "1e-6"))
                cur_h_map, cur_h_valid = _mesh_to_height_maps(mesh, h=hh, w=ww, z_eps=z_eps_map)

                stride = int(os.environ.get("DENSE_MATCH_STRIDE", "1"))
                tau_fb = float(os.environ.get("DENSE_MATCH_TAU_FB", "2.0"))
                max_pts = int(os.environ.get("DENSE_MATCH_MAX_POINTS", "200000"))
                use_bilinear = _as_bool(os.environ.get("DENSE_MATCH_BILINEAR", "1"))

                tau_h_mode = str(os.environ.get("DENSE_MATCH_TAU_H_MODE", "relative")).strip().lower()
                if tau_h_mode not in ("absolute", "relative"):
                    tau_h_mode = "relative"
                tau_h_abs = float(os.environ.get("DENSE_MATCH_TAU_H", "5.0"))
                tau_h_rel = float(os.environ.get("DENSE_MATCH_TAU_H_REL", "0.05"))
                if tau_h_mode == "relative":
                    z_rng = float(np.nanmax(prev_h_map) - np.nanmin(prev_h_map))
                    if (not np.isfinite(z_rng)) or z_rng <= 1e-6:
                        z_rng = float(np.nanmax(cur_h_map) - np.nanmin(cur_h_map))
                    tau_h = float(max(1e-6, tau_h_rel * max(1.0, z_rng)))
                else:
                    tau_h = float(max(1e-6, tau_h_abs))

                dense_out, dense_stats = _dense_match_uv_from_T(
                    T_prev_cur=T_prev_cur,
                    cur_h_map=cur_h_map,
                    cur_valid=cur_h_valid,
                    prev_h_map=prev_h_map,
                    prev_valid=prev_h_valid,
                    stride=stride,
                    tau_h=tau_h,
                    tau_fb=tau_fb,
                    max_points=max_pts,
                    use_bilinear=use_bilinear,
                )

                if dense_out:
                    dense_dir = os.path.join(output_dir, "dense_matches")
                    os.makedirs(dense_dir, exist_ok=True)
                    out_npz = os.path.join(dense_dir, f"dense_matches_prev_{int(idxs[k-1]):06d}_cur_{fi:06d}.npz")
                    np.savez_compressed(
                        out_npz,
                        **dense_out,
                        frame_prev=int(idxs[k - 1]),
                        frame_cur=int(fi),
                        tau_h=float(tau_h),
                        tau_fb=float(tau_fb),
                        stride=int(stride),
                        note="uv are in processed image coordinates (may differ from original if ImageProcessor resized).",
                    )
                    if _as_bool(os.environ.get("PRINT_DENSE_MATCH", "0")):
                        print(
                            f"[DenseMatch][prev={int(idxs[k-1])} cur={fi}] keep={int(dense_stats.get('n_keep',0))}/{int(dense_stats.get('n_src',0))} "
                            f"ratio={dense_stats.get('keep_ratio',0.0):.3f} mean_dh={dense_stats.get('mean_dh',float('nan')):.3f} "
                            f"mean_fb={dense_stats.get('mean_fb',float('nan')):.3f} -> {out_npz}",
                            flush=True,
                        )

                # 更新“上一帧高度图”（用于下一对帧）
                prev_h_map, prev_h_valid, prev_hw = cur_h_map, cur_h_valid, (hh, ww)
            except Exception as e:
                if _as_bool(os.environ.get("PRINT_DENSE_MATCH", "0")):
                    print(f"[DenseMatch][frame={fi}] failed: {e}", flush=True)

        diag_rows.append({
            "frame_idx": int(fi),
            "img": os.path.basename(img_path),
            "pair_fitness": float(reg.fitness),
            "pair_rmse": float(reg.rmse),
            "method": reg.method,
            "init_mode": init_tag,
            "recovered": int(recovered),
            "ref_frame": int(ref_frame),
            "ms_mesh": float(t_mesh),
            "ms_reg": float(t_reg),
            # chessboard pose diagnostics (optional)
            "cb_ok": int(1 if int(fi) in cb_T_c2w else 0),
            "cb_reproj_rmse_px": float(cb_reproj_rmse.get(int(fi), float("nan"))),
            **(gate if isinstance(gate, dict) else {}),
            **mesh_stats,
            "arr_min": (height_stats.get("input_arr_min") if isinstance(height_stats, dict) else ""),
            "arr_max": (height_stats.get("input_arr_max") if isinstance(height_stats, dict) else ""),
            "z_map_mode": (height_stats.get("z_map_mode") if isinstance(height_stats, dict) else ""),
            "z_scale": (height_stats.get("z_scale") if isinstance(height_stats, dict) else ""),
            "z_lin_scale": (height_stats.get("z_lin_scale") if isinstance(height_stats, dict) else ""),
            "z_lin_offset": (height_stats.get("z_lin_offset") if isinstance(height_stats, dict) else ""),
            # dense matching stats (optional)
            "dense_n_src": float(dense_stats.get("n_src", float("nan"))) if isinstance(dense_stats, dict) else "",
            "dense_n_keep": float(dense_stats.get("n_keep", float("nan"))) if isinstance(dense_stats, dict) else "",
            "dense_keep_ratio": float(dense_stats.get("keep_ratio", float("nan"))) if isinstance(dense_stats, dict) else "",
            "dense_mean_dh": float(dense_stats.get("mean_dh", float("nan"))) if isinstance(dense_stats, dict) else "",
            "dense_mean_fb": float(dense_stats.get("mean_fb", float("nan"))) if isinstance(dense_stats, dict) else "",
        })

        print(
            f"[Frame {fi}] mesh {t_mesh:.1f}ms | reg {t_reg:.1f}ms "
            f"| fit={reg.fitness:.3f} rmse={reg.rmse:.4f} method={reg.method}",
            flush=True,
        )

        # 控制台打印 RT（避免刷屏：可用 PRINT_RT_EVERY 控制频率）
        if print_rt and (k % print_rt_every == 0):
            np.set_printoptions(precision=6, suppress=True, linewidth=160)
            prev_i = int(idxs[k - 1])
            if print_rt_mode in ("pairwise", "both"):
                print(f"[RT][pairwise] T_prev<-cur  (prev={prev_i} cur={fi})", flush=True)
                print(T_prev_cur, flush=True)
            if print_rt_mode in ("cumulative", "both"):
                print(f"[RT][cumulative] T_0<-cur   (cur={fi})", flush=True)
                print(T0_cur, flush=True)

        # 默认保存“帧间对齐结果”（到上一帧坐标系）
        if save_pair_results:
            try:
                # 1) 当前帧 mesh 变换到 prev 坐标系
                mesh_in_prev = o3d.geometry.TriangleMesh(mesh)
                mesh_in_prev.transform(T_prev_cur)
                out_mesh = os.path.join(vis_dir, f"pair_prev_{int(idxs[k-1]):06d}_cur_{fi:06d}_cur_in_prev.ply")
                o3d.io.write_triangle_mesh(out_mesh, mesh_in_prev, write_ascii=False)
            except Exception as e:
                print(f"[Warning] save pair mesh failed: {e}")
            try:
                # 2) 叠加点云（prev + cur_aligned）
                pcd_cur_aligned = o3d.geometry.PointCloud(pcd)
                pcd_cur_aligned.transform(T_prev_cur)
                out_pcd = os.path.join(vis_dir, f"pair_prev_{int(idxs[k-1]):06d}_cur_{fi:06d}_overlay_pcd.ply")
                _save_pair_overlay_pcd(out_path=out_pcd, pcd_prev=prev_pcd, pcd_cur_aligned=pcd_cur_aligned)  # type: ignore[arg-type]
            except Exception as e:
                print(f"[Warning] save pair overlay pcd failed: {e}")

        # optional save aligned mesh into frame0 coordinates
        if save_mesh_aligned:
            try:
                mesh_aligned = o3d.geometry.TriangleMesh(mesh)
                mesh_aligned.transform(T0_cur)
                out_ply = os.path.join(vis_dir, f"frame_{fi:06d}_mesh_in_frame0.ply")
                o3d.io.write_triangle_mesh(out_ply, mesh_aligned, write_ascii=False)
            except Exception as e:
                print(f"[Warning] save aligned mesh failed: {e}")

        # update keyframe (periodic + only if quality is good)
        try:
            if (key_pcd_raw is None) or (key_frame_idx is None):
                key_pcd_raw = pcd_raw
                key_frame_idx = int(fi)
                T0_key = T0_cur.copy()
            else:
                if (k % keyframe_interval == 0) and (not _need_recovery(reg.fitness, reg.rmse)):
                    key_pcd_raw = pcd_raw
                    key_frame_idx = int(fi)
                    T0_key = T0_cur.copy()
        except Exception:
            pass

        # shift state
        prev_mesh = mesh
        prev_pcd = pcd
        prev_pcd_raw = pcd_raw
        prev_mesh_stats = mesh_stats
        T0_prev = T0_cur

        if resume_enable and ((k % resume_save_every) == 0):
            _save_progress_checkpoint(last_frame=int(fi))

    # write outputs
    def _save_txt(path: str, mats: dict[int, np.ndarray], title: str) -> None:
        keys = sorted(mats.keys())
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n")
            f.write("# format: frame_idx then 4x4 matrix rows\n")
            for fi in keys:
                f.write(f"frame_idx={fi}\n")
                M = np.asarray(mats[fi], dtype=np.float64)
                for r in range(4):
                    f.write(" ".join([f"{M[r, c]:.10f}" for c in range(4)]) + "\n")
                f.write("\n")

    # chessboard-based physical camera poses (optional)
    if chess_enable and (len(cb_T_c2w) > 0):
        try:
            _save_txt(
                os.path.join(output_dir, "chessboard_camera_poses_c2w_4x4.txt"),
                cb_T_c2w,
                "Chessboard pose (physical): T_c2w (camera -> chessboard/world), units follow CHESSBOARD_SQUARE_SIZE",
            )
            _save_txt(
                os.path.join(output_dir, "chessboard_board_poses_w2c_4x4.txt"),
                cb_T_w2c,
                "Chessboard pose (physical): T_w2c (chessboard/world -> camera), units follow CHESSBOARD_SQUARE_SIZE",
            )
            # pairwise from c2w
            cb_pair: dict[int, np.ndarray] = {}
            keys_cb = sorted(cb_T_c2w.keys())
            for i in range(1, len(keys_cb)):
                f_prev = int(keys_cb[i - 1])
                f_cur = int(keys_cb[i])
                Tp = np.asarray(cb_T_c2w[f_prev], dtype=np.float64)
                Tc = np.asarray(cb_T_c2w[f_cur], dtype=np.float64)
                T_prev_cur = np.linalg.inv(Tp) @ Tc
                cb_pair[f_cur] = T_prev_cur
            if len(keys_cb) > 0:
                cb_pair[int(keys_cb[0])] = np.eye(4, dtype=np.float64)
            _save_txt(
                os.path.join(output_dir, "chessboard_pairwise_4x4.txt"),
                cb_pair,
                "Chessboard pose (physical): T_{t-1<-t} between camera frames derived from T_c2w",
            )
            print("[MeshBetweenRT] chessboard poses saved: chessboard_camera_poses_c2w_4x4.txt", flush=True)
        except Exception as e:
            print(f"[Warning] save chessboard poses failed: {e}", flush=True)

    # save eval csv (mesh->2D mapping quality on chessboard corners)
    if chess_enable and chess_eval_mapping and (len(cb_eval_rows) > 0):
        try:
            eval_csv = os.path.join(output_dir, "dense_match_eval_on_chessboard.csv")
            f, eval_csv_actual = _safe_open_for_write(eval_csv, newline="\n")
            with f:
                fieldnames = [
                    "prev_frame", "cur_frame",
                    "n_prev_corners", "n_cur_corners", "n_cur_h_ok", "n_pred",
                    "err_mean_px", "err_median_px", "err_p95_px",
                    "ok_ratio_tau", "tau_px",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in cb_eval_rows:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
            if eval_csv_actual != eval_csv:
                print(f"[MeshBetweenRT] chessboard eval saved to: {eval_csv_actual}", flush=True)
            else:
                print("[MeshBetweenRT] chessboard eval saved: dense_match_eval_on_chessboard.csv", flush=True)
        except Exception as e:
            print(f"[Warning] save chessboard eval failed: {e}", flush=True)

    # save eval csv (feature matching as corner-role replacement)
    if feat_enable and (len(feat_eval_rows) > 0):
        try:
            eval_csv = os.path.join(output_dir, "feature_match_eval.csv")
            f, eval_csv_actual = _safe_open_for_write(eval_csv, newline="\n")
            with f:
                fieldnames = [
                    "prev_frame", "cur_frame",
                    "method",
                    "n_matches", "n_ok_uvh", "n_inliers",
                    "err_p95_px", "ok_ratio_tau", "tau_px",
                    "eval_p95_max", "eval_ok_min",
                    "note",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in feat_eval_rows:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
            if eval_csv_actual != eval_csv:
                print(f"[MeshBetweenRT] feature eval saved to: {eval_csv_actual}", flush=True)
            else:
                print("[MeshBetweenRT] feature eval saved: feature_match_eval.csv", flush=True)
        except Exception as e:
            print(f"[Warning] save feature eval failed: {e}", flush=True)

    # 形状对齐 RT（推荐使用这些文件名；旧文件名仍会保留以兼容旧脚本）
    _save_txt(os.path.join(output_dir, "shape_rt_pairwise_4x4.txt"), pairwise_T, "Shape RT: T_{t-1<-t} (current->previous) in (u,v,h)")
    _save_txt(os.path.join(output_dir, "shape_rt_cumulative_4x4.txt"), cumulative_T, "Shape RT: T_{0<-t} (current->frame0) in (u,v,h)")
    # backward-compatible legacy names
    _save_txt(os.path.join(output_dir, "rt_pairwise_4x4.txt"), pairwise_T, "LEGACY (same as shape_rt_pairwise_4x4.txt)")
    _save_txt(os.path.join(output_dir, "rt_cumulative_4x4.txt"), cumulative_T, "LEGACY (same as shape_rt_cumulative_4x4.txt)")

    # 额外保存：导出相机位姿(c2w)与相机中心
    # - pose_interp=camera：把累计矩阵 T_{0<-t} 直接当作 c2w
    # - pose_interp=object：把 inverse(T_{0<-t}) 当作 c2w
    try:
        # 注意：下面导出的“camera pose”来自 shape_rt 的解释假设（见 POSE_INTERPRETATION），仅供可视化参考。
        # 为避免误用，额外保存一份带 pseudo 前缀的文件；同时保留旧文件名以兼容旧脚本。
        cam_pose_path = os.path.join(output_dir, "camera_poses_c2w_4x4.txt")
        cam_center_path = os.path.join(output_dir, "camera_centers.txt")
        pseudo_pose_path = os.path.join(output_dir, "pseudo_camera_poses_c2w_4x4.txt")
        pseudo_center_path = os.path.join(output_dir, "pseudo_camera_centers.txt")
        keys = [int(k) for k in idxs if int(k) in cumulative_T]
        def _write_pose_and_centers(pose_path: str, center_path: str, header_prefix: str) -> None:
            with open(pose_path, "w", encoding="utf-8") as f:
                f.write(f"# {header_prefix} camera pose (c2w)\n")
                f.write("# WARNING: derived from shape-alignment RT in (u,v,h). NOT a physical camera pose.\n")
                f.write("# See POSE_INTERPRETATION and meta.json fields.\n")
                for fi in keys:
                    T0_raw = np.asarray(cumulative_T[int(fi)], dtype=np.float64)
                    T0 = np.linalg.inv(T0_raw) if pose_interp == "object" else T0_raw
                    f.write(f"frame_idx={int(fi)}\n")
                    for r in range(4):
                        f.write(" ".join([f"{T0[r, c]:.10f}" for c in range(4)]) + "\n")
                    f.write("\n")
            with open(center_path, "w", encoding="utf-8") as f:
                f.write(f"# {header_prefix} frame_idx cx cy cz\n")
                for fi in keys:
                    T0_raw = np.asarray(cumulative_T[int(fi)], dtype=np.float64)
                    T0 = np.linalg.inv(T0_raw) if pose_interp == "object" else T0_raw
                    c = T0[:3, 3].reshape(3)
                    f.write(f"{int(fi)} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

        _write_pose_and_centers(cam_pose_path, cam_center_path, "LEGACY")
        _write_pose_and_centers(pseudo_pose_path, pseudo_center_path, "PSEUDO")
        print(f"[MeshBetweenRT] camera poses saved: {cam_pose_path}")
        print(f"[MeshBetweenRT] camera centers saved: {cam_center_path}")
        print(f"[MeshBetweenRT] pseudo camera poses saved: {pseudo_pose_path}")
        print(f"[MeshBetweenRT] pseudo camera centers saved: {pseudo_center_path}")
    except Exception as e:
        print(f"[Warning] save camera pose/centers failed: {e}")

    # CSV diag
    diag_csv = os.path.join(output_dir, "diag.csv")
    f, diag_csv_actual = _safe_open_for_write(diag_csv, newline="\n")
    with f:
        fieldnames = [
            "frame_idx", "img", "pair_fitness", "pair_rmse", "method", "ms_mesh", "ms_reg",
            "init_mode", "recovered",
            "ref_frame",
            # chessboard (optional)
            "cb_ok", "cb_reproj_rmse_px",
            # motion gate (jump/self-check)
            "nz_center_dxy", "nz_ratio_mult", "nz_bbox_shift", "gate_jump", "gate_mode",
            # coordinate consistency self-check
            "bbox_min_x", "bbox_min_y", "bbox_min_z",
            "bbox_max_x", "bbox_max_y", "bbox_max_z",
            "center_x", "center_y", "center_z",
            "diag", "z_min", "z_max", "z_mean",
            "nz_ratio",
            "nz_bbox_min_x", "nz_bbox_min_y", "nz_bbox_min_z",
            "nz_bbox_max_x", "nz_bbox_max_y", "nz_bbox_max_z",
            "nz_center_x", "nz_center_y", "nz_center_z",
            "nz_diag", "nz_z_min", "nz_z_max", "nz_z_mean",
            # height-mesh normalization stats (A_t)
            "arr_min", "arr_max", "z_map_mode", "z_scale", "z_lin_scale", "z_lin_offset",
            # dense matching stats (optional)
            "dense_n_src", "dense_n_keep", "dense_keep_ratio", "dense_mean_dh", "dense_mean_fb",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in diag_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    if diag_csv_actual != diag_csv:
        print(f"[MeshBetweenRT] diag saved to: {diag_csv_actual}", flush=True)

    meta = {
        "input_dir": input_dir,
        "coordinate_system": "uvh_heightfield",
        "rt_semantics": "shape_alignment_in_uvh",
        "rt_warning": "RT is computed by ICP on pseudo-3D heightfield mesh (u,v,h). Not a physical camera pose.",
        # chessboard note (optional)
        "chessboard_enable": int(chess_enable),
        "chessboard_try_sizes": [f"{s[0]}x{s[1]}" for s in chess_try_sizes],
        "chessboard_square_size": float(chess_square),
        "chessboard_intrinsics_source": ("provided_json" if bool(chess_intr_path) else "calibrateCamera_or_none"),
        "chessboard_pose_semantics": "solvePnP pose wrt chessboard plane (physical), units follow square_size",
        "num_images": len(image_paths),
        "frames_used": idxs,
        "sample_points": sample_points,
        "voxel_size": voxel_size,
        "icp_method": icp_method,
        "icp_max_corr": icp_max_corr,
        "icp_max_iter": icp_max_iter,
        "use_global_ransac": int(use_global_ransac),
        "ransac_voxel": ransac_voxel,
        "save_mesh_aligned": int(save_mesh_aligned),
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ============================================================
    # PoseGraph global optimization (optional): stabilize trajectory
    # ============================================================
    if pose_graph_opt:
        try:
            keys_pg = [int(k) for k in idxs if int(k) in cumulative_T and int(k) in pg_pcds]
            keys_pg.sort()
            if len(keys_pg) >= 2:
                pg = o3d.pipelines.registration.PoseGraph()
                # nodes: initial poses are current cumulative_T (T_{0<-t})
                for fi in keys_pg:
                    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.asarray(cumulative_T[fi], dtype=np.float64)))

                # edges: consecutive edges (certain)
                for i in range(1, len(keys_pg)):
                    f_prev = keys_pg[i - 1]
                    f_cur = keys_pg[i]
                    # T_prev<-cur = inv(T0_prev) @ T0_cur
                    T_prev_cur = np.linalg.inv(np.asarray(cumulative_T[f_prev], dtype=np.float64)) @ np.asarray(cumulative_T[f_cur], dtype=np.float64)
                    max_corr = float(max(1e-6, pg_voxel * pg_max_corr_mult))
                    try:
                        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                            pg_pcds[f_cur],
                            pg_pcds[f_prev],
                            max_corr,
                            T_prev_cur,
                        )
                    except Exception:
                        info = np.eye(6, dtype=np.float64)
                    pg.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i - 1,
                            i,
                            T_prev_cur,
                            info,
                            uncertain=False,
                        )
                    )

                # loop edges: keyframe recovery references (uncertain)
                for row in diag_rows:
                    try:
                        fi = int(row.get("frame_idx", -1))
                        ref = int(row.get("ref_frame", -1))
                        if fi < 0 or ref < 0 or fi == ref:
                            continue
                        if fi not in keys_pg or ref not in keys_pg:
                            continue
                        # Only add when recovery happened (strong loop closure signal)
                        if int(row.get("recovered", 0)) != 1:
                            continue
                        idx_i = keys_pg.index(ref)
                        idx_j = keys_pg.index(fi)
                        # T_ref<-fi = inv(T0_ref) @ T0_fi
                        T_ref_fi = np.linalg.inv(np.asarray(cumulative_T[ref], dtype=np.float64)) @ np.asarray(cumulative_T[fi], dtype=np.float64)
                        max_corr = float(max(1e-6, pg_voxel * pg_max_corr_mult))
                        try:
                            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                                pg_pcds[fi],
                                pg_pcds[ref],
                                max_corr,
                                T_ref_fi,
                            )
                        except Exception:
                            info = np.eye(6, dtype=np.float64)
                        pg.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                idx_i,
                                idx_j,
                                T_ref_fi,
                                info,
                                uncertain=True,
                            )
                        )
                    except Exception:
                        continue

                option = o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=float(max(1e-6, pg_voxel * pg_max_corr_mult)),
                    edge_prune_threshold=0.25,
                    reference_node=0,
                )
                o3d.pipelines.registration.global_optimization(
                    pg,
                    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                    option,
                )

                # save optimized cumulative poses
                optimized: dict[int, np.ndarray] = {}
                for i, fi in enumerate(keys_pg):
                    optimized[fi] = np.asarray(pg.nodes[i].pose, dtype=np.float64)
                _save_txt(
                    os.path.join(output_dir, "shape_rt_cumulative_optimized_4x4.txt"),
                    optimized,
                    "Shape RT (optimized): T_{0<-t} after pose-graph optimization",
                )
                print("[MeshBetweenRT] pose-graph optimized poses saved: shape_rt_cumulative_optimized_4x4.txt", flush=True)
        except Exception as e:
            print(f"[Warning] pose-graph optimization failed: {e}", flush=True)

    print("=" * 80)
    print(f"[MeshBetweenRT] done. outputs in: {output_dir}")
    print(f"[MeshBetweenRT] shape pairwise: shape_rt_pairwise_4x4.txt")
    print(f"[MeshBetweenRT] shape cumulative: shape_rt_cumulative_4x4.txt")
    print(f"[MeshBetweenRT] (legacy) pairwise: rt_pairwise_4x4.txt")
    print(f"[MeshBetweenRT] (legacy) cumulative: rt_cumulative_4x4.txt")
    print(f"[MeshBetweenRT] diag: diag.csv")
    print(f"[MeshBetweenRT] pair vis (mesh/pcd): {os.path.join(output_dir,'vis')}")
    print("=" * 80)

    if vis_open3d:
        try:
            # 可视化：绘制相机位姿（坐标轴/视锥/轨迹）
            # - pose_interp=camera：T_{0<-t} 直接当作 c2w
            # - pose_interp=object：对 T_{0<-t} 取逆再当作 c2w
            geoms: list[o3d.geometry.Geometry] = []
            # choose coordinate frame size
            frame_size = 50.0
            if vis_frame_size_s != "auto":
                try:
                    frame_size = float(vis_frame_size_s)
                except Exception:
                    frame_size = 50.0

            # reload from processor cache if possible
            mesh0 = None
            try:
                if hasattr(processor, "_mesh_cache") and int(idxs[0]) in processor._mesh_cache:
                    mesh0 = processor._mesh_cache[int(idxs[0])]
            except Exception:
                mesh0 = None
            if mesh0 is not None:
                m0 = o3d.geometry.TriangleMesh(mesh0)
                m0.paint_uniform_color([0.7, 0.7, 0.7])
                geoms.append(m0)
                # auto size by bbox diagonal
                if vis_frame_size_s == "auto":
                    try:
                        bb = m0.get_axis_aligned_bounding_box()
                        diag = float(np.linalg.norm(np.asarray(bb.get_extent(), dtype=np.float64)))
                        if np.isfinite(diag) and diag > 1e-6:
                            frame_size = max(1e-3, 0.15 * diag)
                    except Exception:
                        pass

            show_n = int(os.environ.get("VIS_MAX_MESHES", "8"))
            show_n = max(1, show_n)
            show_frames = idxs[:show_n]
            traj_frames = idxs if vis_traj_use_all else show_frames

            # 绘制“相机位姿”：坐标轴/视锥 + 轨迹 (in frame0 coords)
            if vis_show_rt_frames:
                origins: list[np.ndarray] = []
                for ii, fidx in enumerate(traj_frames):
                    if ii % vis_rt_every != 0:
                        continue
                    try:
                        T0_raw = np.asarray(cumulative_T.get(int(fidx), np.eye(4)), dtype=np.float64)
                        T0 = np.linalg.inv(T0_raw) if pose_interp == "object" else T0_raw
                        fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(frame_size))
                        fr.transform(T0)
                        geoms.append(fr)
                        if vis_show_camera_frustum:
                            frus = _make_camera_frustum_lineset(scale=float(frame_size) * 0.8)
                            frus.transform(T0)
                            geoms.append(frus)
                        origins.append(T0[:3, 3].copy())
                    except Exception:
                        pass
                if len(origins) >= 2:
                    pts = np.stack(origins, axis=0).astype(np.float64)
                    lines = np.stack([np.arange(len(origins) - 1), np.arange(1, len(origins))], axis=1).astype(np.int32)
                    ls = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(pts),
                        lines=o3d.utility.Vector2iVector(lines),
                    )
                    ls.colors = o3d.utility.Vector3dVector(
                        np.tile(np.array([[0.1, 0.7, 1.0]], dtype=np.float64), (lines.shape[0], 1))
                    )
                    geoms.append(ls)
                    pcd_pts = o3d.geometry.PointCloud()
                    pcd_pts.points = o3d.utility.Vector3dVector(pts)
                    pcd_pts.colors = o3d.utility.Vector3dVector(
                        np.tile(np.array([[1.0, 0.9, 0.1]], dtype=np.float64), (pts.shape[0], 1))
                    )
                    geoms.append(pcd_pts)
            for fi in show_frames[1:]:
                try:
                    if hasattr(processor, "_mesh_cache") and int(fi) in processor._mesh_cache:
                        m = o3d.geometry.TriangleMesh(processor._mesh_cache[int(fi)])
                        Tm_raw = np.asarray(cumulative_T.get(int(fi), np.eye(4)), dtype=np.float64)
                        Tm = np.linalg.inv(Tm_raw) if pose_interp == "object" else Tm_raw
                        m.transform(Tm)
                        m.paint_uniform_color([0.2, 0.8, 0.2])
                        geoms.append(m)
                except Exception:
                    pass
            o3d.visualization.draw_geometries(geoms, window_name="Camera Poses from Mesh-to-Mesh RT (aligned to frame0)")
        except Exception as e:
            print(f"[Warning] Open3D vis failed: {e}")

    try:
        if log_f is not None:
            log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

