"""
Inertial branch: learned IMU-to-camera relative-pose regression.
"""

from src.imu.models import (
    IMUCameraPoseGRU6D,
    IMUCameraPoseMLP6D,
    PoseScalerRT,
)
from src.imu.losses import (
    PoseGeodesicLoss,
    AdaptivePoseGeodesicLoss,
)
from src.imu.dataset import (
    IMUCameraDataset,
    prepare_data_splits,
)

__all__ = [
    "IMUCameraPoseGRU6D",
    "IMUCameraPoseMLP6D",
    "PoseScalerRT",
    "PoseGeodesicLoss",
    "AdaptivePoseGeodesicLoss",
    "IMUCameraDataset",
    "prepare_data_splits",
]
