# Run from the repository root:
#
# python -m gym_pybullet_drones.predictiveMonitoringControl.run_camera_oracle \
#     --gui True --plot True --duration_sec 180

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync

from .perception import ColorObstaclePerception, PerceptionConfig
from .virtual_camera import CameraConfig, VirtualCamera

from .state_estimator import (
    CameraStateEstimator,
    StateEstimatorConfig,
)


DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 180
DEFAULT_OUTPUT_FOLDER = "results/camera_oracle_test"

START = np.array([-1.20, 0.00], dtype=float)
GOAL = np.array([4.20, 0.00], dtype=float)

GOAL_TOLERANCE = 0.12

NOMINAL_SPEED = 0.18
AVOID_SPEED = 0.16
SLOW_SPEED = 0.10

SAFETY_MARGIN = 0.18
ACTIVATION_DISTANCE = 0.54
DEACTIVATION_DISTANCE = 0.05
GUIDE_LATERAL_EXTRA = 0.24
REPULSION_GAIN = 1.20

# name, x, y, radius, colour, desired bypass side
OBSTACLE_DATA = [
    (
        "O1",
        -0.10,
        0.18,
        0.20,
        [1.0, 0.0, 0.0, 1.0],
        "RIGHT",
    ),
    (
        "O2",
        1.15,
        -0.18,
        0.20,
        [0.0, 0.0, 1.0, 1.0],
        "LEFT",
    ),
    (
        "O3",
        2.40,
        0.18,
        0.20,
        [1.0, 1.0, 0.0, 1.0],
        "RIGHT",
    ),
    (
        "O4",
        3.65,
        -0.18,
        0.20,
        [0.0, 1.0, 0.0, 1.0],
        "LEFT",
    ),
]


def create_output_directory(base_folder: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_directory = os.path.join(
        base_folder,
        f"run_{timestamp}",
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    return output_directory


def create_visible_obstacles(
    client_id: int,
) -> list[int]:
    body_ids: list[int] = []
    obstacle_height = 0.50

    for _, x, y, radius, colour, _ in OBSTACLE_DATA:
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=obstacle_height,
            physicsClientId=client_id,
        )

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=obstacle_height,
            rgbaColor=colour,
            physicsClientId=client_id,
        )

        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[
                x,
                y,
                obstacle_height / 2.0,
            ],
            physicsClientId=client_id,
        )

        body_ids.append(body_id)

    return body_ids


def calculate_clearances(
    position_xy: np.ndarray,
) -> tuple[float, float, str]:
    minimum_physical_clearance = np.inf
    minimum_effective_clearance = np.inf
    nearest_obstacle_name = "none"

    for name, x, y, radius, _, _ in OBSTACLE_DATA:
        obstacle_position = np.array(
            [x, y],
            dtype=float,
        )

        centre_distance = np.linalg.norm(
            position_xy - obstacle_position
        )

        physical_clearance = centre_distance - radius

        effective_clearance = (
            centre_distance
            - radius
            - SAFETY_MARGIN
        )

        if physical_clearance < minimum_physical_clearance:
            minimum_physical_clearance = physical_clearance
            minimum_effective_clearance = effective_clearance
            nearest_obstacle_name = name

    return (
        float(minimum_physical_clearance),
        float(minimum_effective_clearance),
        nearest_obstacle_name,
    )


class TemporaryAlternatingController:
    """
    Temporary geometry-based controller.

    Navigation still uses known obstacle coordinates.

    The camera perception module is only observed and validated
    in this version. Later, this geometry controller will be
    replaced by the camera-based safety oracle.
    """

    def __init__(self) -> None:
        self.active_obstacle_index: int | None = None

    def _find_next_obstacle(
        self,
        position_xy: np.ndarray,
    ) -> int | None:
        candidate_index: int | None = None
        smallest_forward_distance = np.inf

        for index, (_, x, _, radius, _, _) in enumerate(
            OBSTACLE_DATA
        ):
            forward_distance = (
                x
                + radius
                + DEACTIVATION_DISTANCE
                - position_xy[0]
            )

            if (
                forward_distance >= 0.0
                and forward_distance < smallest_forward_distance
            ):
                smallest_forward_distance = forward_distance
                candidate_index = index

        return candidate_index

    def _build_guide_point(
        self,
        obstacle_index: int,
    ) -> np.ndarray:
        (
            _,
            obstacle_x,
            obstacle_y,
            radius,
            _,
            bypass_side,
        ) = OBSTACLE_DATA[obstacle_index]

        lateral_offset = (
            radius
            + SAFETY_MARGIN
            + GUIDE_LATERAL_EXTRA
        )

        guide_x = (
            obstacle_x
            + radius
            + 0.10
        )

        if bypass_side == "LEFT":
            guide_y = obstacle_y + lateral_offset
        else:
            guide_y = obstacle_y - lateral_offset

        return np.array(
            [guide_x, guide_y],
            dtype=float,
        )

    def command(
        self,
        position_xy: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        to_goal = GOAL - position_xy
        goal_distance = np.linalg.norm(to_goal)

        (
            physical_clearance,
            effective_clearance,
            nearest_obstacle,
        ) = calculate_clearances(position_xy)

        if goal_distance < GOAL_TOLERANCE:
            return (
                np.zeros(4, dtype=float),
                {
                    "physical_clearance": physical_clearance,
                    "effective_clearance": effective_clearance,
                    "nearest_obstacle": nearest_obstacle,
                    "active_obstacle": "none",
                    "speed": 0.0,
                    "mode": "GOAL_REACHED",
                },
            )

        if self.active_obstacle_index is None:
            next_obstacle_index = self._find_next_obstacle(
                position_xy
            )

            if next_obstacle_index is not None:
                (
                    _,
                    obstacle_x,
                    obstacle_y,
                    _,
                    _,
                    _,
                ) = OBSTACLE_DATA[next_obstacle_index]

                obstacle_position = np.array(
                    [obstacle_x, obstacle_y],
                    dtype=float,
                )

                centre_distance = np.linalg.norm(
                    position_xy - obstacle_position
                )

                if centre_distance < ACTIVATION_DISTANCE:
                    self.active_obstacle_index = (
                        next_obstacle_index
                    )

        if self.active_obstacle_index is not None:
            (
                active_name,
                active_x,
                active_y,
                active_radius,
                _,
                bypass_side,
            ) = OBSTACLE_DATA[
                self.active_obstacle_index
            ]

            obstacle_position = np.array(
                [active_x, active_y],
                dtype=float,
            )

            passing_boundary_x = (
                active_x
                + active_radius
                + DEACTIVATION_DISTANCE
            )

            obstacle_passed = (
                position_xy[0] > passing_boundary_x
                and physical_clearance > 0.10
            )

            if obstacle_passed:
                print(
                    f"Passed {active_name} at "
                    f"x={position_xy[0]:.3f}"
                )

                self.active_obstacle_index = None

                direction = to_goal / max(
                    goal_distance,
                    1e-9,
                )

                return (
                    np.array(
                        [
                            direction[0],
                            direction[1],
                            0.0,
                            NOMINAL_SPEED,
                        ],
                        dtype=float,
                    ),
                    {
                        "physical_clearance": physical_clearance,
                        "effective_clearance": effective_clearance,
                        "nearest_obstacle": nearest_obstacle,
                        "active_obstacle": "none",
                        "speed": NOMINAL_SPEED,
                        "mode": "NOMINAL",
                    },
                )

            guide_point = self._build_guide_point(
                self.active_obstacle_index
            )

            to_guide = guide_point - position_xy
            guide_distance = np.linalg.norm(to_guide)

            if guide_distance > 1e-9:
                guide_direction = (
                    to_guide / guide_distance
                )
            else:
                guide_direction = np.array(
                    [1.0, 0.0],
                    dtype=float,
                )

            away_vector = (
                position_xy - obstacle_position
            )

            away_distance = np.linalg.norm(
                away_vector
            )

            repulsion = np.zeros(
                2,
                dtype=float,
            )

            repulsion_influence_radius = (
                active_radius
                + SAFETY_MARGIN
                + 0.16
            )

            if (
                away_distance > 1e-9
                and away_distance
                < repulsion_influence_radius
            ):
                proximity = (
                    repulsion_influence_radius
                    - away_distance
                ) / 0.16

                proximity = float(
                    np.clip(
                        proximity,
                        0.0,
                        1.0,
                    )
                )

                repulsion = (
                    REPULSION_GAIN
                    * proximity
                    * away_vector
                    / away_distance
                )

            direction = guide_direction + repulsion

            hard_safety_threshold = (
                active_radius
                + SAFETY_MARGIN
                + 0.05
            )

            if (
                away_distance < hard_safety_threshold
                and away_distance > 1e-9
            ):
                outward_direction = (
                    away_vector / away_distance
                )

                penetration_ratio = (
                    hard_safety_threshold
                    - away_distance
                ) / hard_safety_threshold

                penetration_ratio = float(
                    np.clip(
                        penetration_ratio,
                        0.0,
                        1.0,
                    )
                )

                direction = (
                    0.35 * guide_direction
                    + 1.80
                    * penetration_ratio
                    * outward_direction
                )

            direction_norm = np.linalg.norm(
                direction
            )

            if direction_norm < 1e-9:
                direction = np.array(
                    [1.0, 0.0],
                    dtype=float,
                )
            else:
                direction = (
                    direction / direction_norm
                )

            if effective_clearance < 0.04:
                speed = SLOW_SPEED
            elif effective_clearance < 0.10:
                speed = 0.13
            else:
                speed = AVOID_SPEED

            return (
                np.array(
                    [
                        direction[0],
                        direction[1],
                        0.0,
                        speed,
                    ],
                    dtype=float,
                ),
                {
                    "physical_clearance": physical_clearance,
                    "effective_clearance": effective_clearance,
                    "nearest_obstacle": nearest_obstacle,
                    "active_obstacle": active_name,
                    "speed": speed,
                    "mode": f"AVOID_{bypass_side}",
                },
            )

        direction = to_goal / max(
            goal_distance,
            1e-9,
        )

        return (
            np.array(
                [
                    direction[0],
                    direction[1],
                    0.0,
                    NOMINAL_SPEED,
                ],
                dtype=float,
            ),
            {
                "physical_clearance": physical_clearance,
                "effective_clearance": effective_clearance,
                "nearest_obstacle": nearest_obstacle,
                "active_obstacle": "none",
                "speed": NOMINAL_SPEED,
                "mode": "NOMINAL",
            },
        )


def save_trajectory_csv(
    trajectory_log: list[dict],
    output_directory: str,
) -> None:
    csv_path = os.path.join(
        output_directory,
        "camera_oracle_trajectory.csv",
    )

    columns = [
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "physical_clearance_m",
        "effective_clearance_m",
        "nearest_obstacle",
        "active_obstacle",
        "commanded_speed_mps",
        "control_mode",
        "detected_obstacles",
        "nearest_detected_distance_m",
        "nearest_detected_bearing_deg",
        "nearest_detected_confidence",
    ]

    with open(
        csv_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            ",".join(columns) + "\n"
        )

        for row in trajectory_log:
            values = [
                f"{row['time_s']:.6f}",
                f"{row['x_m']:.6f}",
                f"{row['y_m']:.6f}",
                f"{row['z_m']:.6f}",
                f"{row['physical_clearance_m']:.6f}",
                f"{row['effective_clearance_m']:.6f}",
                str(row["nearest_obstacle"]),
                str(row["active_obstacle"]),
                f"{row['commanded_speed_mps']:.6f}",
                str(row["control_mode"]),
                str(row["detected_obstacles"]),
                f"{row['nearest_detected_distance_m']:.6f}",
                f"{row['nearest_detected_bearing_deg']:.6f}",
                f"{row['nearest_detected_confidence']:.6f}",
            ]

            file.write(
                ",".join(values) + "\n"
            )

    print(
        f"Saved trajectory CSV: {csv_path}"
    )


def plot_2d_trajectory(
    trajectory_log: list[dict],
    output_directory: str,
    show_plot: bool,
) -> None:
    if not trajectory_log:
        print(
            "No trajectory samples were recorded."
        )
        return

    x_values = np.array(
        [
            row["x_m"]
            for row in trajectory_log
        ],
        dtype=float,
    )

    y_values = np.array(
        [
            row["y_m"]
            for row in trajectory_log
        ],
        dtype=float,
    )

    figure, axis = plt.subplots(
        figsize=(10.5, 4.8)
    )

    axis.plot(
        x_values,
        y_values,
        linewidth=2.2,
        label="Drone trajectory",
    )

    axis.scatter(
        [x_values[0]],
        [y_values[0]],
        marker="o",
        s=80,
        label="Start",
        zorder=5,
    )

    axis.scatter(
        [GOAL[0]],
        [GOAL[1]],
        marker="*",
        s=180,
        label="Goal",
        zorder=5,
    )

    for (
        name,
        x,
        y,
        radius,
        _,
        bypass_side,
    ) in OBSTACLE_DATA:
        physical_obstacle = plt.Circle(
            (x, y),
            radius,
            fill=True,
            alpha=0.30,
        )

        safety_boundary = plt.Circle(
            (x, y),
            radius + SAFETY_MARGIN,
            fill=False,
            linestyle="--",
            linewidth=1.8,
        )

        axis.add_patch(
            physical_obstacle
        )

        axis.add_patch(
            safety_boundary
        )

        axis.text(
            x,
            y,
            f"{name}\n{bypass_side}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_xlabel(
        "x position [m]"
    )

    axis.set_ylabel(
        "y position [m]"
    )

    axis.set_title(
        "Geometry-Controlled UAV Path with Camera Perception"
    )

    axis.axis("equal")
    axis.grid(True)
    axis.legend(loc="upper right")

    axis.set_xlim(-1.45, 4.45)
    axis.set_ylim(-0.85, 0.85)

    figure.tight_layout()

    png_path = os.path.join(
        output_directory,
        "camera_oracle_trajectory_2d.png",
    )

    pdf_path = os.path.join(
        output_directory,
        "camera_oracle_trajectory_2d.pdf",
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print(
        f"Saved 2D trajectory plot: {png_path}"
    )

    print(
        f"Saved 2D trajectory PDF: {pdf_path}"
    )

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def main(
    gui: bool = True,
    plot: bool = True,
    duration_sec: int = DEFAULT_DURATION_SEC,
    simulation_freq_hz: int = DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz: int = DEFAULT_CONTROL_FREQ_HZ,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
) -> None:
    output_directory = create_output_directory(
        output_folder
    )

    print(
        f"Saving outputs to: "
        f"{output_directory}"
    )

    initial_positions = np.array(
        [
            [
                START[0],
                START[1],
                0.30,
            ]
        ],
        dtype=float,
    )

    initial_orientations = np.array(
        [[0.0, 0.0, 0.0]],
        dtype=float,
    )

    env = VelocityAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=initial_positions,
        initial_rpys=initial_orientations,
        physics=Physics.PYB,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    visible_obstacle_ids = (
        create_visible_obstacles(
            env.CLIENT
        )
    )

    print(
        "Visible obstacle IDs:",
        visible_obstacle_ids,
    )

    print(
        "Drone body ID:",
        env.DRONE_IDS[0],
    )

    print(
        "Geometry navigation with camera "
        "perception started."
    )

    camera = VirtualCamera(
        client_id=env.CLIENT,
        drone_body_id=env.DRONE_IDS[0],
        config=CameraConfig(
            width=320,
            height=240,
            fov_deg=70.0,
            near_plane=0.02,
            far_plane=5.0,
            offset_body=(
                0.05,
                0.0,
                0.025,
            ),
            pitch_deg=-8.0,
        ),
    )

    perception = ColorObstaclePerception(
        config=PerceptionConfig(
            image_width=320,
            image_height=240,
            horizontal_fov_deg=70.0,
            obstacle_diameter_m=0.40,
        )
    )
    
    state_estimator = CameraStateEstimator(
        config=StateEstimatorConfig(
            minimum_confidence=0.15,
            maximum_distance_m=5.0,
            maximum_frame_age_s=0.50,
        )
    )


    controller = (
        TemporaryAlternatingController()
    )

    action = np.zeros(
        (1, 4),
        dtype=float,
    )

    # Send a valid movement command before the first env.step().
    action[0, :], _ = controller.command(
        START.copy()
    )

    trajectory_log: list[dict] = []

    # Latest detections remain available between camera frames.
    observations = []
    estimated_obstacles = []


    number_of_steps = int(
        duration_sec * env.CTRL_FREQ
    )

    start_time = time.time()

    try:
        for step in range(number_of_steps):
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            simulation_time = (
                step / env.CTRL_FREQ
            )

            drone_position = np.asarray(
                observation[0][0:3],
                dtype=float,
            )

            position_xy = (
                drone_position[:2]
            )

            # Geometry controller still determines movement.
            action[0, :], diagnostics = (
                controller.command(
                    position_xy
                )
            )

            # Run camera and perception at about 16 FPS.
            if step % 3 == 0:
                camera_output = (
                    camera.capture()
                )

                rgb_frame = camera_output[
                    "rgb"
                ]

                observations = perception.process(
                    rgb_frame=rgb_frame,
                    timestamp_s=simulation_time,
                )

                estimated_obstacles = state_estimator.update(
                    observations=observations,
                    current_time_s=simulation_time,
                )

                bgr_frame = perception.draw_detections(
                    rgb_frame=rgb_frame,
                    observations=observations,
                )

                if step == 0:
                    first_frame_path = os.path.join(
                        output_directory,
                        "first_virtual_camera_frame.png",
                    )

                    saved = cv2.imwrite(
                        first_frame_path,
                        bgr_frame,
                    )

                    if saved:
                        print(
                            "Saved first camera frame: "
                            f"{first_frame_path}"
                        )
                    else:
                        print(
                            "Warning: first camera "
                            "frame was not saved."
                        )

                information_text = (
                    f"Mode: {diagnostics['mode']} | "
                    f"Active: "
                    f"{diagnostics['active_obstacle']} | "
                    f"Clearance: "
                    f"{diagnostics['physical_clearance']:.2f} m | "
                    f"Detected: {len(observations)}"
                )

                cv2.putText(
                    bgr_frame,
                    information_text,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "Simulated onboard camera",
                    bgr_frame,
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    print(
                        "Stopped by user."
                    )
                    break

            if observations:
                nearest_detection = min(
                    observations,
                    key=lambda item: (
                        item.estimated_distance_m
                    ),
                )

                detected_names = "|".join(
                    observation_item.obstacle_name
                    for observation_item
                    in observations
                )

                detected_distance = (
                    nearest_detection
                    .estimated_distance_m
                )

                detected_bearing_deg = float(
                    np.rad2deg(
                        nearest_detection
                        .bearing_rad
                    )
                )

                detected_confidence = (
                    nearest_detection.confidence
                )
            else:
                detected_names = "none"
                detected_distance = float("nan")
                detected_bearing_deg = float("nan")
                detected_confidence = 0.0

            trajectory_log.append(
                {
                    "time_s": simulation_time,
                    "x_m": float(
                        drone_position[0]
                    ),
                    "y_m": float(
                        drone_position[1]
                    ),
                    "z_m": float(
                        drone_position[2]
                    ),
                    "physical_clearance_m": diagnostics[
                        "physical_clearance"
                    ],
                    "effective_clearance_m": diagnostics[
                        "effective_clearance"
                    ],
                    "nearest_obstacle": diagnostics[
                        "nearest_obstacle"
                    ],
                    "active_obstacle": diagnostics[
                        "active_obstacle"
                    ],
                    "commanded_speed_mps": diagnostics[
                        "speed"
                    ],
                    "control_mode": diagnostics[
                        "mode"
                    ],
                    "detected_obstacles": (
                        detected_names
                    ),
                    "nearest_detected_distance_m": (
                        detected_distance
                    ),
                    "nearest_detected_bearing_deg": (
                        detected_bearing_deg
                    ),
                    "nearest_detected_confidence": (
                        detected_confidence
                    ),
                }
            )

            if step % env.CTRL_FREQ == 0:
                print(
                    f"t={simulation_time:.1f} s | "
                    f"x={drone_position[0]:.3f}, "
                    f"y={drone_position[1]:.3f} | "
                    f"mode={diagnostics['mode']} | "
                    f"active="
                    f"{diagnostics['active_obstacle']} | "
                    f"clearance="
                    f"{diagnostics['physical_clearance']:.3f} m"
                )

                if observations:
                    for detected in observations:
                        print(
                            f"  Detected "
                            f"{detected.obstacle_name} | "
                            f"distance="
                            f"{detected.estimated_distance_m:.2f} m | "
                            f"bearing="
                            f"{np.rad2deg(detected.bearing_rad):.1f} deg | "
                            f"confidence="
                            f"{detected.confidence:.2f}"
                        )
                else:
                    print(
                        "  No coloured obstacle detected."
                    )

                if estimated_obstacles:
                    for obstacle in estimated_obstacles:
                        print(
                            f"  Estimated {obstacle.obstacle_name} | "
                            f"forward={obstacle.forward_m:.2f} m | "
                            f"lateral={obstacle.lateral_m:.2f} m | "
                            f"distance={obstacle.distance_m:.2f} m | "
                            f"confidence={obstacle.confidence:.2f}"
                        )
                else:
                    print("  No valid estimated obstacle state.")    

            if (
                np.linalg.norm(
                    position_xy - GOAL
                )
                < GOAL_TOLERANCE
            ):
                print(
                    f"Reached final goal at "
                    f"t={simulation_time:.2f} s."
                )
                break

            if terminated or truncated:
                print(
                    "Environment terminated at "
                    f"t={simulation_time:.2f} s."
                )
                break

            if gui:
                sync(
                    step,
                    start_time,
                    env.CTRL_TIMESTEP,
                )

    except KeyboardInterrupt:
        print(
            "\nSimulation interrupted."
        )

    finally:
        cv2.destroyAllWindows()
        env.close()

        save_trajectory_csv(
            trajectory_log,
            output_directory,
        )

        plot_2d_trajectory(
            trajectory_log,
            output_directory,
            show_plot=plot,
        )

        if trajectory_log:
            minimum_physical_clearance = min(
                row["physical_clearance_m"]
                for row in trajectory_log
            )

            minimum_effective_clearance = min(
                row["effective_clearance_m"]
                for row in trajectory_log
            )

            final_row = trajectory_log[-1]

            print("\nRun summary:")

            print(
                "Minimum physical clearance: "
                f"{minimum_physical_clearance:.3f} m"
            )

            print(
                "Minimum effective clearance: "
                f"{minimum_effective_clearance:.3f} m"
            )

            print(
                f"Final position: "
                f"x={final_row['x_m']:.3f}, "
                f"y={final_row['y_m']:.3f}"
            )

        print(
            "Simulation finished."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Geometry-controlled UAV navigation "
            "with onboard colour perception"
        )
    )

    parser.add_argument(
        "--gui",
        default=True,
        type=str2bool,
    )

    parser.add_argument(
        "--plot",
        default=True,
        type=str2bool,
    )

    parser.add_argument(
        "--duration_sec",
        default=DEFAULT_DURATION_SEC,
        type=int,
    )

    parser.add_argument(
        "--simulation_freq_hz",
        default=DEFAULT_SIMULATION_FREQ_HZ,
        type=int,
    )

    parser.add_argument(
        "--control_freq_hz",
        default=DEFAULT_CONTROL_FREQ_HZ,
        type=int,
    )

    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
    )

    arguments = parser.parse_args()

    main(
        **vars(arguments)
    )