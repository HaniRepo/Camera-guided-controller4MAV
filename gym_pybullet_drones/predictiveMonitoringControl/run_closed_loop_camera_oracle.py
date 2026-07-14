# Run from repository root:
# python -m gym_pybullet_drones.predictiveMonitoringControl.run_closed_loop_camera_oracle --gui True --plot True --duration_sec 180

from __future__ import annotations

import argparse
import csv
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
from .safety_oracle import CameraSafetyOracle, SafetyOracleConfig
from .state_estimator import CameraStateEstimator, StateEstimatorConfig
from .virtual_camera import CameraConfig, VirtualCamera


DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 180
DEFAULT_OUTPUT_FOLDER = "results/camera_oracle_test"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WINDOW_NAME = "Simulated onboard camera"

START = np.array([-1.20, 0.00], dtype=float)
GOAL = np.array([4.20, 0.00], dtype=float)
GOAL_TOLERANCE = 0.12

NOMINAL_SPEED = 0.18
SAFETY_MARGIN = 0.18

OBSTACLE_DATA = [
    ("O1", -0.10,  0.18, 0.20, [1.0, 0.0, 0.0, 1.0], "RIGHT"),
    ("O2",  1.15, -0.18, 0.20, [0.0, 0.0, 1.0, 1.0], "LEFT"),
    ("O3",  2.40,  0.18, 0.20, [1.0, 1.0, 0.0, 1.0], "RIGHT"),
    ("O4",  3.65, -0.18, 0.20, [0.0, 1.0, 0.0, 1.0], "LEFT"),
]


def create_output_directory(base_folder: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(base_folder, f"run_{timestamp}")
    os.makedirs(output_directory, exist_ok=True)
    return output_directory


def create_visible_obstacles(client_id: int) -> list[int]:
    obstacle_height = 0.50
    ids: list[int] = []

    for _, x, y, radius, colour, _ in OBSTACLE_DATA:
        collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=obstacle_height,
            physicsClientId=client_id,
        )
        visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=obstacle_height,
            rgbaColor=colour,
            physicsClientId=client_id,
        )
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, obstacle_height / 2.0],
            physicsClientId=client_id,
        )
        ids.append(body)

    return ids


def calculate_clearances(position_xy: np.ndarray) -> tuple[float, float, str]:
    min_physical = float("inf")
    min_effective = float("inf")
    nearest = "none"

    for name, x, y, radius, _, _ in OBSTACLE_DATA:
        distance = float(
            np.linalg.norm(position_xy - np.array([x, y], dtype=float))
        )
        physical = distance - radius
        effective = physical - SAFETY_MARGIN

        if physical < min_physical:
            min_physical = physical
            min_effective = effective
            nearest = name

    return min_physical, min_effective, nearest


def normalize_direction(vector_xy: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector_xy))
    if norm < 1e-9:
        return np.array([0.0, 0.0], dtype=float)
    return np.asarray(vector_xy, dtype=float) / norm


def draw_compact_detections(rgb_frame: np.ndarray, observations: list) -> np.ndarray:
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    colour_map = {
        "O1_RED": (0, 0, 255),
        "O2_BLUE": (255, 0, 0),
        "O3_YELLOW": (0, 255, 255),
        "O4_GREEN": (0, 255, 0),
    }

    for obs in observations:
        x, y, w, h = obs.bounding_box
        colour = colour_map.get(obs.obstacle_name, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.circle(
            frame,
            (int(obs.center_x_px), int(obs.center_y_px)),
            4,
            colour,
            -1,
        )
        cv2.putText(
            frame,
            f"{obs.obstacle_name} {obs.estimated_distance_m:.2f}m",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            1,
            cv2.LINE_AA,
        )

    return frame


def build_display(
    bgr_frame: np.ndarray,
    decision,
    physical_clearance: float,
    detection_count: int,
) -> np.ndarray:
    display = cv2.copyMakeBorder(
        bgr_frame,
        76,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )

    cv2.putText(
        display,
        f"Mode: {decision.mode}    Active: {decision.active_obstacle}",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        (
            f"Risk: {decision.risk_score:.2f}    "
            f"Camera distance: {decision.estimated_distance_m:.2f} m    "
            f"True clearance: {physical_clearance:.2f} m    "
            f"Detected: {detection_count}"
        ),
        (14, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return display


def save_csv(log: list[dict], output_directory: str) -> None:
    path = os.path.join(
        output_directory,
        "closed_loop_camera_oracle_trajectory.csv",
    )
    if not log:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(log[0].keys()))
        writer.writeheader()
        writer.writerows(log)

    print(f"Saved trajectory CSV: {path}")


def plot_trajectory(
    log: list[dict],
    output_directory: str,
    show_plot: bool,
) -> None:
    if not log:
        print("No trajectory samples were recorded.")
        return

    x_values = np.array([float(row["x_m"]) for row in log])
    y_values = np.array([float(row["y_m"]) for row in log])

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(x_values, y_values, linewidth=2.2, label="Camera-oracle trajectory")
    ax.scatter([x_values[0]], [y_values[0]], s=80, label="Start", zorder=5)
    ax.scatter([GOAL[0]], [GOAL[1]], marker="*", s=180, label="Goal", zorder=5)

    for name, x, y, radius, _, side in OBSTACLE_DATA:
        ax.add_patch(plt.Circle((x, y), radius, fill=True, alpha=0.30))
        ax.add_patch(
            plt.Circle(
                (x, y),
                radius + SAFETY_MARGIN,
                fill=False,
                linestyle="--",
                linewidth=1.8,
            )
        )
        ax.text(
            x,
            y,
            f"{name}\n{side}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.set_title("Closed-Loop Camera Safety Oracle")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.set_xlim(-1.45, 4.45)
    ax.set_ylim(-0.95, 0.95)
    fig.tight_layout()

    png_path = os.path.join(
        output_directory,
        "closed_loop_camera_oracle_trajectory_2d.png",
    )
    pdf_path = os.path.join(
        output_directory,
        "closed_loop_camera_oracle_trajectory_2d.pdf",
    )

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved 2D trajectory plot: {png_path}")
    print(f"Saved 2D trajectory PDF: {pdf_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main(
    gui: bool = True,
    plot: bool = True,
    duration_sec: int = DEFAULT_DURATION_SEC,
    simulation_freq_hz: int = DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz: int = DEFAULT_CONTROL_FREQ_HZ,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
) -> None:
    output_directory = create_output_directory(output_folder)
    print(f"Saving outputs to: {output_directory}")

    env = VelocityAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[START[0], START[1], 0.30]], dtype=float),
        initial_rpys=np.array([[0.0, 0.0, 0.0]], dtype=float),
        physics=Physics.PYB,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    obstacle_ids = create_visible_obstacles(env.CLIENT)
    print("Visible obstacle IDs:", obstacle_ids)
    print("Drone body ID:", env.DRONE_IDS[0])
    print("Closed-loop camera-oracle test started.")

    camera = VirtualCamera(
        client_id=env.CLIENT,
        drone_body_id=env.DRONE_IDS[0],
        config=CameraConfig(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fov_deg=70.0,
            near_plane=0.02,
            far_plane=5.0,
            offset_body=(0.05, 0.0, 0.025),
            pitch_deg=-8.0,
        ),
    )

    perception = ColorObstaclePerception(
        config=PerceptionConfig(
            image_width=CAMERA_WIDTH,
            image_height=CAMERA_HEIGHT,
            horizontal_fov_deg=70.0,
            obstacle_diameter_m=0.40,
            minimum_contour_area_px=300.0,
            minimum_width_px=10.0,
            morphology_kernel_size=5,
        )
    )

    state_estimator = CameraStateEstimator(
        config=StateEstimatorConfig(
            minimum_confidence=0.15,
            maximum_distance_m=5.0,
            maximum_frame_age_s=0.50,
        )
    )

    oracle = CameraSafetyOracle(
        config=SafetyOracleConfig(
            nominal_speed=0.18,
            avoid_speed=0.17,
            slow_speed=0.13,
            warning_distance_m=1.20,
            critical_distance_m=0.65,
            lateral_command_strength=1.45,
            critical_lateral_strength=1.70,
            minimum_confidence=0.15,
            maximum_frame_age_s=0.50,
            release_after_missing_frames=8,
        )
    )

    initial_direction = normalize_direction(GOAL - START)
    action = np.array(
        [[initial_direction[0], initial_direction[1], 0.0, NOMINAL_SPEED]],
        dtype=float,
    )

    observations = []
    estimated_obstacles = []
    decision = oracle.evaluate(
        nominal_direction_xy=initial_direction,
        estimated_obstacles=[],
    )

    trajectory_log: list[dict] = []
    number_of_steps = int(duration_sec * env.CTRL_FREQ)
    camera_interval_steps = 3
    start_time = time.time()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 780)

    try:
        for step in range(number_of_steps):
            observation, reward, terminated, truncated, info = env.step(action)

            simulation_time = step / env.CTRL_FREQ
            drone_position = np.asarray(observation[0][0:3], dtype=float)
            position_xy = drone_position[:2]

            physical_clearance, effective_clearance, nearest_true = (
                calculate_clearances(position_xy)
            )

            to_goal = GOAL - position_xy
            goal_distance = float(np.linalg.norm(to_goal))
            nominal_direction_xy = normalize_direction(to_goal)

            if step % camera_interval_steps == 0:
                rgb_frame = camera.capture()["rgb"]

                observations = perception.process(
                    rgb_frame=rgb_frame,
                    timestamp_s=simulation_time,
                )

                estimated_obstacles = state_estimator.update(
                    observations=observations,
                    current_time_s=simulation_time,
                )

                decision = oracle.evaluate(
                    nominal_direction_xy=nominal_direction_xy,
                    estimated_obstacles=estimated_obstacles,
                )

                action[0, :] = decision.safe_command

                bgr_frame = draw_compact_detections(
                    rgb_frame,
                    observations,
                )

                display_frame = build_display(
                    bgr_frame,
                    decision,
                    physical_clearance,
                    len(observations),
                )

                if step == 0:
                    first_frame_path = os.path.join(
                        output_directory,
                        "first_virtual_camera_frame.png",
                    )
                    cv2.imwrite(first_frame_path, display_frame)
                    print(f"Saved first camera frame: {first_frame_path}")

                cv2.imshow(WINDOW_NAME, display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            if observations:
                nearest_detection = min(
                    observations,
                    key=lambda item: item.estimated_distance_m,
                )
                detected_names = "|".join(
                    item.obstacle_name for item in observations
                )
                detected_distance = float(
                    nearest_detection.estimated_distance_m
                )
                detected_bearing = float(
                    np.rad2deg(nearest_detection.bearing_rad)
                )
                detected_confidence = float(
                    nearest_detection.confidence
                )
            else:
                detected_names = "none"
                detected_distance = float("nan")
                detected_bearing = float("nan")
                detected_confidence = 0.0

            trajectory_log.append(
                {
                    "time_s": f"{simulation_time:.6f}",
                    "x_m": f"{drone_position[0]:.6f}",
                    "y_m": f"{drone_position[1]:.6f}",
                    "z_m": f"{drone_position[2]:.6f}",
                    "physical_clearance_m": f"{physical_clearance:.6f}",
                    "effective_clearance_m": f"{effective_clearance:.6f}",
                    "nearest_true_obstacle": nearest_true,
                    "active_camera_obstacle": decision.active_obstacle,
                    "commanded_speed_mps": f"{decision.safe_command[3]:.6f}",
                    "control_mode": decision.mode,
                    "risk_score": f"{decision.risk_score:.6f}",
                    "oracle_reason": decision.reason,
                    "detected_obstacles": detected_names,
                    "nearest_detected_distance_m": f"{detected_distance:.6f}",
                    "nearest_detected_bearing_deg": f"{detected_bearing:.6f}",
                    "nearest_detected_confidence": f"{detected_confidence:.6f}",
                }
            )

            if step % env.CTRL_FREQ == 0:
                print(
                    f"t={simulation_time:.1f} s | "
                    f"x={drone_position[0]:.3f}, "
                    f"y={drone_position[1]:.3f} | "
                    f"mode={decision.mode} | "
                    f"active={decision.active_obstacle} | "
                    f"true_clearance={physical_clearance:.3f} m | "
                    f"camera_distance={decision.estimated_distance_m:.3f} m | "
                    f"risk={decision.risk_score:.2f}"
                )

            if goal_distance < GOAL_TOLERANCE:
                print(f"Reached final goal at t={simulation_time:.2f} s.")
                break

            if terminated or truncated:
                print(f"Environment terminated at t={simulation_time:.2f} s.")
                break

            if gui:
                sync(step, start_time, env.CTRL_TIMESTEP)

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")

    finally:
        cv2.destroyAllWindows()
        env.close()

        save_csv(trajectory_log, output_directory)
        plot_trajectory(trajectory_log, output_directory, plot)

        if trajectory_log:
            min_physical = min(
                float(row["physical_clearance_m"])
                for row in trajectory_log
            )
            min_effective = min(
                float(row["effective_clearance_m"])
                for row in trajectory_log
            )
            final_row = trajectory_log[-1]

            print("\nRun summary:")
            print(f"Minimum physical clearance: {min_physical:.3f} m")
            print(f"Minimum effective clearance: {min_effective:.3f} m")
            print(
                "Final position: "
                f"x={float(final_row['x_m']):.3f}, "
                f"y={float(final_row['y_m']):.3f}"
            )

        print("Simulation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop UAV navigation using virtual-camera "
            "perception and a persistent camera safety oracle"
        )
    )

    parser.add_argument("--gui", default=True, type=str2bool)
    parser.add_argument("--plot", default=True, type=str2bool)
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

    args = parser.parse_args()
    main(**vars(args))