"""
Utilities package for pose estimation, geometry, metrics, and data I/O.
"""

from src.utils.geometry import (
    rot6d_to_rotmat_torch,
    rot6d_to_rotmat_np,
    geodesic_distance_so3,
    invert_se3,
    compose_se3,
)
from src.utils.metrics import (
    compute_rpe,
    compute_ate_aligned,
    aggregate_stats,
)
from src.utils.data_io import (
    read_tum_trajectory,
    save_tum_trajectory,
    load_4x4_matrix_file,
    save_4x4_matrix_file,
)

__all__ = [
    "rot6d_to_rotmat_torch",
    "rot6d_to_rotmat_np",
    "geodesic_distance_so3",
    "invert_se3",
    "compose_se3",
    "compute_rpe",
    "compute_ate_aligned",
    "aggregate_stats",
    "read_tum_trajectory",
    "save_tum_trajectory",
    "load_4x4_matrix_file",
    "save_4x4_matrix_file",
]
