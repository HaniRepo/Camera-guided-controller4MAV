from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from state_estimator import EstimatedObstacleState


BYPASS_SIDE_BY_OBSTACLE = {
    "O1_RED": "RIGHT",
    "O2_BLUE": "LEFT",
    "O3_YELLOW": "RIGHT",
    "O4_GREEN": "LEFT",
}


@dataclass
class SafetyOracleConfig:
    nominal_speed: float = 0.18
    avoid_speed: float = 0.17
    slow_speed: float = 0.13

    warning_distance_m: float = 1.20
    critical_distance_m: float = 0.65

    lateral_command_strength: float = 1.45
    critical_lateral_strength: float = 1.70

    minimum_confidence: float = 0.15
    maximum_frame_age_s: float = 0.50
    release_after_missing_frames: int = 8


@dataclass
class OracleDecision:
    safe_command: np.ndarray
    mode: str
    active_obstacle: str
    estimated_distance_m: float
    estimated_bearing_rad: float
    risk_score: float
    reason: str


class CameraSafetyOracle:
    def __init__(
        self,
        config: SafetyOracleConfig | None = None,
    ) -> None:
        self.config = config or SafetyOracleConfig()

        self.active_obstacle_name: str | None = None
        self.active_bypass_side: str | None = None
        self.missing_active_frames = 0
        self.last_distance = float("inf")
        self.last_bearing = 0.0

    @staticmethod
    def _normalize(direction_xy: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(direction_xy))
        if norm < 1e-9:
            return np.array([1.0, 0.0], dtype=float)
        return np.asarray(direction_xy, dtype=float) / norm

    def _command(
        self,
        direction_xy: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        direction = self._normalize(direction_xy)

        return np.array(
            [
                direction[0],
                direction[1],
                0.0,
                float(speed),
            ],
            dtype=float,
        )

    def _risk(self, distance: float) -> float:
        denominator = (
            self.config.warning_distance_m
            - self.config.critical_distance_m
        )

        if denominator <= 1e-9:
            return 1.0

        return float(
            np.clip(
                (
                    self.config.warning_distance_m
                    - distance
                )
                / denominator,
                0.0,
                1.0,
            )
        )

    def _avoid_direction(
        self,
        nominal_direction_xy: np.ndarray,
        bypass_side: str,
        critical: bool,
    ) -> np.ndarray:
        nominal = self._normalize(
            nominal_direction_xy
        )
        lateral_sign = (
            1.0 if bypass_side == "LEFT" else -1.0
        )

        if critical:
            direction = np.array(
                [
                    0.45,
                    lateral_sign
                    * self.config.critical_lateral_strength,
                ],
                dtype=float,
            )
        else:
            direction = np.array(
                [
                    max(0.65, float(nominal[0])),
                    (
                        0.20 * float(nominal[1])
                        + lateral_sign
                        * self.config.lateral_command_strength
                    ),
                ],
                dtype=float,
            )

        return self._normalize(direction)

    def _nominal(
        self,
        nominal_direction_xy: np.ndarray,
        active_obstacle: str = "none",
        distance: float = float("inf"),
        bearing: float = 0.0,
        reason: str = "NO_VALID_CAMERA_OBSTACLE",
    ) -> OracleDecision:
        return OracleDecision(
            safe_command=self._command(
                nominal_direction_xy,
                self.config.nominal_speed,
            ),
            mode="NOMINAL",
            active_obstacle=active_obstacle,
            estimated_distance_m=distance,
            estimated_bearing_rad=bearing,
            risk_score=0.0,
            reason=reason,
        )

    def evaluate(
        self,
        nominal_direction_xy: np.ndarray,
        estimated_obstacles: list[EstimatedObstacleState],
    ) -> OracleDecision:
        valid = [
            obstacle
            for obstacle in estimated_obstacles
            if (
                obstacle.confidence
                >= self.config.minimum_confidence
                and obstacle.frame_age_s
                <= self.config.maximum_frame_age_s
                and obstacle.forward_m > 0.0
                and np.isfinite(obstacle.distance_m)
            )
        ]

        if self.active_obstacle_name is not None:
            matches = [
                obstacle
                for obstacle in valid
                if obstacle.obstacle_name
                == self.active_obstacle_name
            ]

            if matches:
                obstacle = min(
                    matches,
                    key=lambda item: item.distance_m,
                )
                self.missing_active_frames = 0
                self.last_distance = float(
                    obstacle.distance_m
                )
                self.last_bearing = float(
                    obstacle.bearing_rad
                )
            else:
                self.missing_active_frames += 1

                if (
                    self.missing_active_frames
                    < self.config.release_after_missing_frames
                ):
                    side = (
                        self.active_bypass_side
                        or "RIGHT"
                    )
                    direction = self._avoid_direction(
                        nominal_direction_xy,
                        side,
                        critical=False,
                    )

                    return OracleDecision(
                        safe_command=self._command(
                            direction,
                            self.config.avoid_speed,
                        ),
                        mode=f"PASSING_{side}",
                        active_obstacle=(
                            self.active_obstacle_name
                        ),
                        estimated_distance_m=(
                            self.last_distance
                        ),
                        estimated_bearing_rad=(
                            self.last_bearing
                        ),
                        risk_score=0.0,
                        reason=(
                            "ACTIVE_OBSTACLE_TEMPORARILY_MISSING"
                        ),
                    )

                self.active_obstacle_name = None
                self.active_bypass_side = None
                self.missing_active_frames = 0
                self.last_distance = float("inf")
                self.last_bearing = 0.0

        if self.active_obstacle_name is None:
            if not valid:
                return self._nominal(
                    nominal_direction_xy
                )

            nearest = min(
                valid,
                key=lambda item: item.distance_m,
            )

            if (
                nearest.distance_m
                >= self.config.warning_distance_m
            ):
                return self._nominal(
                    nominal_direction_xy,
                    active_obstacle=(
                        nearest.obstacle_name
                    ),
                    distance=float(
                        nearest.distance_m
                    ),
                    bearing=float(
                        nearest.bearing_rad
                    ),
                    reason=(
                        "OBSTACLE_OUTSIDE_WARNING_DISTANCE"
                    ),
                )

            self.active_obstacle_name = (
                nearest.obstacle_name
            )
            self.active_bypass_side = (
                BYPASS_SIDE_BY_OBSTACLE.get(
                    nearest.obstacle_name,
                    "RIGHT",
                )
            )
            self.missing_active_frames = 0
            self.last_distance = float(
                nearest.distance_m
            )
            self.last_bearing = float(
                nearest.bearing_rad
            )

        matches = [
            obstacle
            for obstacle in valid
            if obstacle.obstacle_name
            == self.active_obstacle_name
        ]

        if matches:
            obstacle = min(
                matches,
                key=lambda item: item.distance_m,
            )
            distance = float(
                obstacle.distance_m
            )
            bearing = float(
                obstacle.bearing_rad
            )
            self.last_distance = distance
            self.last_bearing = bearing
        else:
            distance = self.last_distance
            bearing = self.last_bearing

        side = self.active_bypass_side or "RIGHT"
        critical = (
            distance
            <= self.config.critical_distance_m
        )

        direction = self._avoid_direction(
            nominal_direction_xy,
            side,
            critical,
        )

        if critical:
            speed = self.config.slow_speed
            mode = f"CRITICAL_AVOID_{side}"
            reason = "CAMERA_CRITICAL_DISTANCE"
        else:
            speed = self.config.avoid_speed
            mode = f"AVOID_{side}"
            reason = "CAMERA_WARNING_DISTANCE"

        return OracleDecision(
            safe_command=self._command(
                direction,
                speed,
            ),
            mode=mode,
            active_obstacle=(
                self.active_obstacle_name
                or "none"
            ),
            estimated_distance_m=distance,
            estimated_bearing_rad=bearing,
            risk_score=self._risk(distance),
            reason=reason,
        )