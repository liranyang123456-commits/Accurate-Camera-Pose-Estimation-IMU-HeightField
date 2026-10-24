#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Height-field tracker used for every visual experiment of the paper (one implementation, one configuration).

Front end : Scharr gradient magnitude -> empirical q-quantile truncation -> normalization to [0, z_max]
            -> standardization to (mu_ref, sigma_ref) -> lifting to (u, v, z) on a stride-s lattice.
Registration: three-level point-to-plane ICP (Open3D), voxel sizes (4, 2, 1) px, correspondence radii
            (15, 8, 4) px, iteration caps (50, 30, 20); consecutive frames, no keyframes or pose graph.
Configuration: image width 640 px, q = 0.68, s = 2, at most 8000 points per frame.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.visual.height_field import (  # noqa: E402
    compute_gradient_magnitude,
    lift_to_pointcloud_uvh,
    stabilize_height_distribution,
    truncate_gradient_percentile,
)
from src.visual.mesh_registration import register_pointclouds_multiscale  # noqa: E402

CFG = {"width": 640, "q": 0.68, "step": 2, "max_pts": 8000}


def resize_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def static_mask(frames: list[np.ndarray], thr: int = 12, erode_px: int = 15) -> np.ndarray:
    """Image-only mask of the endoscope field of view (excludes black vignette corners and their rim)."""
    mx = np.max(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]), axis=0)
    m = (mx > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))


def height_cloud(img_bgr: np.ndarray, mask: np.ndarray | None = None, cfg: dict = CFG) -> np.ndarray:
    img = resize_width(img_bgr, cfg["width"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    z, _ = truncate_gradient_percentile(compute_gradient_magnitude(gray), percentile=cfg["q"])
    z = stabilize_height_distribution(z)
    if mask is not None:
        if mask.shape[:2] != z.shape:
            mask = cv2.resize(mask, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_NEAREST)
        z = np.where(mask > 0, z, 0.0)
    pts = lift_to_pointcloud_uvh(z, step=cfg["step"])
    if pts.shape[0] > cfg["max_pts"]:
        pts = pts[np.linspace(0, pts.shape[0] - 1, cfg["max_pts"]).astype(int)]
    return pts


def track(images, mask=None, cfg: dict = CFG, timing: bool = False):
    """Chain consecutive registrations into camera-to-reference poses T_{0<-t} (pixel units)."""
    poses, fits = [np.eye(4)], []
    t_front, t_reg = [], []
    prev = None
    for k, img in enumerate(images):
        t0 = time.perf_counter()
        cloud = height_cloud(img, mask, cfg)
        t1 = time.perf_counter()
        if prev is not None:
            if len(cloud) >= 30 and len(prev) >= 30:
                T, fit, _ = register_pointclouds_multiscale(cloud, prev)
            else:
                T, fit = np.eye(4), 0.0
            poses.append(poses[-1] @ T)
            fits.append(float(fit))
            t_reg.append(time.perf_counter() - t1)
        t_front.append(t1 - t0)
        prev = cloud
    out = {"poses": np.stack(poses), "fitness": np.asarray(fits)}
    if timing:
        out["t_front"] = np.asarray(t_front)
        out["t_reg"] = np.asarray(t_reg)
    return out
