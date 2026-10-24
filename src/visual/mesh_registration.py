"""
Multi-scale Point-to-Plane ICP Registration on Pseudo-3D Height Fields.
Registers consecutive or keyframe elevation point clouds.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import open3d as o3d


def register_pointclouds_multiscale(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    init_transform: Optional[np.ndarray] = None,
    voxel_scales: Tuple[float, ...] = (4.0, 2.0, 1.0),
    max_iterations: Tuple[int, ...] = (50, 30, 20),
    max_correspondence_distances: Tuple[float, ...] = (15.0, 8.0, 4.0),
) -> Tuple[np.ndarray, float, float]:
    """
    Perform multi-scale Point-to-Plane ICP registration.

    Args:
        source_pts: (N, 3) numpy array for source point cloud (frame t)
        target_pts: (M, 3) numpy array for target point cloud (frame t-1)
        init_transform: 4x4 initial SE(3) transformation matrix
        voxel_scales: Decreasing voxel sizes for coarse-to-fine registration
        max_iterations: Maximum iterations per scale
        max_correspondence_distances: Search radius per scale

    Returns:
        T_opt: 4x4 optimized SE(3) transformation
        fitness: Inlier correspondence overlap ratio (0.0 to 1.0)
        inlier_rmse: RMSE of inlier correspondences
    """
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(source_pts)

    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(target_pts)

    if init_transform is None:
        current_transform = np.eye(4, dtype=np.float64)
    else:
        current_transform = np.asarray(init_transform, dtype=np.float64).copy()

    fitness = 0.0
    inlier_rmse = 999.0

    for voxel_size, max_iter, max_corr in zip(voxel_scales, max_iterations, max_correspondence_distances):
        # Downsample
        if voxel_size > 0:
            src_down = pcd_src.voxel_down_sample(voxel_size)
            tgt_down = pcd_tgt.voxel_down_sample(voxel_size)
        else:
            src_down = pcd_src
            tgt_down = pcd_tgt

        # Estimate normals on target for point-to-plane objective
        tgt_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr * 1.5, max_nn=30)
        )

        reg = o3d.pipelines.registration.registration_icp(
            src_down,
            tgt_down,
            max_corr,
            current_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

        current_transform = reg.transformation
        fitness = reg.fitness
        inlier_rmse = reg.inlier_rmse

    return current_transform, float(fitness), float(inlier_rmse)
