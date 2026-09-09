"""
Visual branch: Structural Gradient Height Field Front-End, ICP, Quality Gating, and Pose Graph.
"""

from src.visual.height_field import (
    compute_gradient_magnitude,
    truncate_gradient_percentile,
    stabilize_height_distribution,
    lift_to_pointcloud_uvh,
)
from src.visual.quality_gate import (
    classify_motion_state,
    check_corner_reprojection_gate,
)
from src.visual.mesh_registration import (
    register_pointclouds_multiscale,
)
from src.visual.pose_graph import (
    VisualPoseGraph,
)

__all__ = [
    "compute_gradient_magnitude",
    "truncate_gradient_percentile",
    "stabilize_height_distribution",
    "lift_to_pointcloud_uvh",
    "classify_motion_state",
    "check_corner_reprojection_gate",
    "register_pointclouds_multiscale",
    "VisualPoseGraph",
]
