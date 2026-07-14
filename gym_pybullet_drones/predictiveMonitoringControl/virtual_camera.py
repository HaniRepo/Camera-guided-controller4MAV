from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p


@dataclass
class CameraConfig:
    width: int = 320
    height: int = 240
    fov_deg: float = 70.0
    near_plane: float = 0.02
    far_plane: float = 5.0

    # Camera location relative to the drone body.
    # x = forward, y = left, z = upward
    offset_body: tuple[float, float, float] = (0.05, 0.0, 0.025)

    # Slight downward tilt.
    pitch_deg: float = -8.0

    # Distance used only to construct the camera target point.
    target_distance: float = 1.0


class VirtualCamera:
    """Forward-facing camera rigidly attached to a PyBullet drone."""

    def __init__(
        self,
        client_id: int,
        drone_body_id: int,
        config: Optional[CameraConfig] = None,
    ) -> None:
        self.client_id = client_id
        self.drone_body_id = drone_body_id
        self.config = config or CameraConfig()

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.fov_deg,
            aspect=self.config.width / self.config.height,
            nearVal=self.config.near_plane,
            farVal=self.config.far_plane,
        )

    def _camera_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return camera eye, target and up vectors in world coordinates.
        """
        body_position, body_quaternion = p.getBasePositionAndOrientation(
            self.drone_body_id,
            physicsClientId=self.client_id,
        )

        body_position = np.asarray(body_position, dtype=float)

        rotation_matrix = np.asarray(
            p.getMatrixFromQuaternion(body_quaternion),
            dtype=float,
        ).reshape(3, 3)

        offset_body = np.asarray(self.config.offset_body, dtype=float)
        camera_position = body_position + rotation_matrix @ offset_body

        pitch_rad = np.deg2rad(self.config.pitch_deg)

        # Camera looks along the drone's positive x-axis.
        forward_body = np.array(
            [
                np.cos(pitch_rad),
                0.0,
                np.sin(pitch_rad),
            ],
            dtype=float,
        )

        up_body = np.array(
            [
                -np.sin(pitch_rad),
                0.0,
                np.cos(pitch_rad),
            ],
            dtype=float,
        )

        forward_world = rotation_matrix @ forward_body
        up_world = rotation_matrix @ up_body

        camera_target = (
            camera_position
            + self.config.target_distance * forward_world
        )

        return camera_position, camera_target, up_world

    def capture(self) -> dict[str, np.ndarray]:
        """
        Capture RGB, depth and segmentation images.

        The controller should later use RGB only.
        Depth and segmentation are retained for debugging and evaluation.
        """
        camera_position, camera_target, up_vector = self._camera_pose()

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=up_vector.tolist(),
        )

        width, height, rgba, depth, segmentation = p.getCameraImage(
            width=self.config.width,
            height=self.config.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )

        rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
        rgb = rgba[:, :, :3]

        depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
        segmentation = np.asarray(
            segmentation,
            dtype=np.int32,
        ).reshape(height, width)

        return {
            "rgb": rgb,
            "depth": depth,
            "segmentation": segmentation,
            "camera_position": camera_position,
            "camera_target": camera_target,
        }