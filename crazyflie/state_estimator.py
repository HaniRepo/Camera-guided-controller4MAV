from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception import ObstacleObservation


@dataclass
class EstimatedObstacleState:
    obstacle_name: str

    # Position relative to the drone/camera.
    forward_m: float
    lateral_m: float

    distance_m: float
    bearing_rad: float

    confidence: float
    observation_timestamp_s: float
    frame_age_s: float


@dataclass
class StateEstimatorConfig:
    minimum_confidence: float = 0.15
    maximum_distance_m: float = 5.0
    maximum_frame_age_s: float = 0.50


class CameraStateEstimator:
    """
    Convert image-derived obstacle observations into a simple
    camera-relative planar obstacle state.

    Coordinate convention:
        forward_m > 0: obstacle is in front of the drone
        lateral_m > 0: obstacle is to the left
        lateral_m < 0: obstacle is to the right
    """

    def __init__(
        self,
        config: StateEstimatorConfig | None = None,
    ) -> None:
        self.config = config or StateEstimatorConfig()

    def update(
        self,
        observations: list[ObstacleObservation],
        current_time_s: float,
    ) -> list[EstimatedObstacleState]:
        estimated_obstacles: list[EstimatedObstacleState] = []

        for observation in observations:
            frame_age_s = max(
                0.0,
                current_time_s - observation.timestamp_s,
            )

            if observation.confidence < self.config.minimum_confidence:
                continue

            if observation.estimated_distance_m > self.config.maximum_distance_m:
                continue

            if frame_age_s > self.config.maximum_frame_age_s:
                continue

            distance_m = observation.estimated_distance_m
            bearing_rad = observation.bearing_rad

            forward_m = distance_m * np.cos(bearing_rad)
            lateral_m = distance_m * np.sin(bearing_rad)

            estimated_obstacles.append(
                EstimatedObstacleState(
                    obstacle_name=observation.obstacle_name,
                    forward_m=float(forward_m),
                    lateral_m=float(lateral_m),
                    distance_m=float(distance_m),
                    bearing_rad=float(bearing_rad),
                    confidence=float(observation.confidence),
                    observation_timestamp_s=float(
                        observation.timestamp_s
                    ),
                    frame_age_s=float(frame_age_s),
                )
            )

        estimated_obstacles.sort(
            key=lambda obstacle: obstacle.distance_m
        )

        return estimated_obstacles