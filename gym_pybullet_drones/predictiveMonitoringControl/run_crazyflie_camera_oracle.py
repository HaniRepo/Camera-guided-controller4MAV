from __future__ import annotations

import csv
import logging
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cflib.crtp
import matplotlib.pyplot as plt
import pandas as pd

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


# ============================================================
# Crazyflie configuration
# ============================================================

URI = "radio://0/80/2M"

# Keep the Crazyflie close to the floor.
# Increase to 0.18-0.20 m if ground effect causes instability.
FLIGHT_HEIGHT_M = 0.22

FORWARD_SPEED_MPS = 0.07
LATERAL_SPEED_MPS = 0.05
VERTICAL_SPEED_MPS = 0.10

COMMAND_PAUSE_S = 0.40


# ============================================================
# Experimental map
# ============================================================

MAP_LENGTH_X_M = 1.00
MAP_WIDTH_Y_M = 0.50

# MotionCommander coordinates:
# +X = forward
# -X = backward
# +Y = left
# -Y = right
# +Z = up
# -Z = down

START_X_M = 0.08
START_Y_M = 0.25

OBSTACLES = [
    {
        "name": "O1",
        "cx": 0.36,
        "cy": 0.25,
        "radius": 0.055,
    },
    {
        "name": "O2",
        "cx": 0.67,
        "cy": 0.25,
        "radius": 0.055,
    },
]

SAFE_CLEARANCE_M = 0.08
WARNING_CLEARANCE_M = 0.13


# ============================================================
# Straight-motion route
# ============================================================

# Start:
#   x = 0.08 m
#   y = 0.25 m
#
# Route:
#   1) forward 0.12 m
#   2) right   0.16 m
#   3) forward 0.62 m
#   4) left    0.16 m
#   5) forward 0.10 m
#   6) land
#
# Expected landing point:
#   x = 0.92 m
#   y = 0.25 m

APPROACH_DISTANCE_M = 0.12
RIGHT_SHIFT_M = 0.16
PASS_DISTANCE_M = 0.62
LEFT_SHIFT_M = 0.16
FINAL_FORWARD_M = 0.10


# ============================================================
# Results
# ============================================================

RUN_DIRECTORY = (
    Path("results")
    / "crazyflie_two_obstacle_straight_motion"
    / datetime.now().strftime("run_%Y%m%d_%H%M%S")
)

CSV_PATH = RUN_DIRECTORY / "two_obstacle_straight_log.csv"
TRAJECTORY_PATH = RUN_DIRECTORY / "two_obstacle_trajectory.png"
CLEARANCE_PATH = RUN_DIRECTORY / "two_obstacle_clearance.png"
RISK_PATH = RUN_DIRECTORY / "two_obstacle_risk.png"
COMMANDS_PATH = RUN_DIRECTORY / "two_obstacle_commands.png"

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")


# ============================================================
# Command-derived pose
# ============================================================

@dataclass
class Pose2D:
    x: float
    y: float


pose = Pose2D(
    x=START_X_M,
    y=START_Y_M,
)

logs: list[dict] = []
experiment_start_s = 0.0


def elapsed_s() -> float:
    return time.monotonic() - experiment_start_s


def nearest_obstacle_clearance(
    x_m: float,
    y_m: float,
) -> tuple[str, float]:
    nearest_name = "none"
    minimum_clearance = float("inf")

    for obstacle in OBSTACLES:
        centre_distance = math.hypot(
            x_m - obstacle["cx"],
            y_m - obstacle["cy"],
        )

        clearance = (
            centre_distance
            - obstacle["radius"]
        )

        if clearance < minimum_clearance:
            minimum_clearance = clearance
            nearest_name = obstacle["name"]

    return nearest_name, minimum_clearance


def risk_from_clearance(
    clearance_m: float,
) -> float:
    if clearance_m >= WARNING_CLEARANCE_M:
        return 0.0

    if clearance_m <= SAFE_CLEARANCE_M:
        return 1.0

    return (
        WARNING_CLEARANCE_M
        - clearance_m
    ) / (
        WARNING_CLEARANCE_M
        - SAFE_CLEARANCE_M
    )


def append_log(
    timestamp_s: float,
    label: str,
    event: str,
    x_m: float,
    y_m: float,
    vx_mps: float,
    vy_mps: float,
) -> None:
    obstacle_name, clearance_m = (
        nearest_obstacle_clearance(
            x_m,
            y_m,
        )
    )

    logs.append(
        {
            "time_s": timestamp_s,
            "event": event,
            "label": label,
            "x_est_m": x_m,
            "y_est_m": y_m,
            "vx_mps": vx_mps,
            "vy_mps": vy_mps,
            "nearest_obstacle": obstacle_name,
            "clearance_m": clearance_m,
            "risk": risk_from_clearance(
                clearance_m
            ),
            "monitor_active": int(
                clearance_m
                < WARNING_CLEARANCE_M
            ),
        }
    )


# ============================================================
# Route validation
# ============================================================

def validate_route() -> None:
    route_vertices = [
        (
            START_X_M,
            START_Y_M,
        ),
        (
            START_X_M + APPROACH_DISTANCE_M,
            START_Y_M,
        ),
        (
            START_X_M + APPROACH_DISTANCE_M,
            START_Y_M - RIGHT_SHIFT_M,
        ),
        (
            START_X_M
            + APPROACH_DISTANCE_M
            + PASS_DISTANCE_M,
            START_Y_M - RIGHT_SHIFT_M,
        ),
        (
            START_X_M
            + APPROACH_DISTANCE_M
            + PASS_DISTANCE_M,
            START_Y_M
            - RIGHT_SHIFT_M
            + LEFT_SHIFT_M,
        ),
        (
            START_X_M
            + APPROACH_DISTANCE_M
            + PASS_DISTANCE_M
            + FINAL_FORWARD_M,
            START_Y_M
            - RIGHT_SHIFT_M
            + LEFT_SHIFT_M,
        ),
    ]

    for index, (x_m, y_m) in enumerate(
        route_vertices
    ):
        if not (
            0.0 <= x_m <= MAP_LENGTH_X_M
            and
            0.0 <= y_m <= MAP_WIDTH_Y_M
        ):
            raise ValueError(
                f"Route vertex {index} is outside "
                f"the map: x={x_m:.3f}, y={y_m:.3f}"
            )

    corridor_y = (
        START_Y_M - RIGHT_SHIFT_M
    )

    for obstacle in OBSTACLES:
        clearance = (
            abs(
                corridor_y
                - obstacle["cy"]
            )
            - obstacle["radius"]
        )

        if clearance < SAFE_CLEARANCE_M:
            raise ValueError(
                f"Route is unsafe near "
                f"{obstacle['name']}: "
                f"clearance={clearance:.3f} m"
            )

    landing_x, landing_y = route_vertices[-1]

    print("Route validation passed.")
    print(
        f"Expected landing point: "
        f"x={landing_x:.2f} m, "
        f"y={landing_y:.2f} m"
    )


# ============================================================
# Dense logging for each straight segment
# ============================================================

def record_linear_segment(
    start_x_m: float,
    start_y_m: float,
    delta_x_m: float,
    delta_y_m: float,
    speed_mps: float,
    command_start_s: float,
    label: str,
    vx_mps: float,
    vy_mps: float,
) -> None:
    distance_m = math.hypot(
        delta_x_m,
        delta_y_m,
    )

    duration_s = distance_m / speed_mps

    sample_count = max(
        2,
        int(duration_s / 0.05),
    )

    for sample_index in range(
        sample_count + 1
    ):
        fraction = (
            sample_index / sample_count
        )

        append_log(
            timestamp_s=(
                command_start_s
                + fraction * duration_s
            ),
            label=label,
            event="MOVING",
            x_m=(
                start_x_m
                + fraction * delta_x_m
            ),
            y_m=(
                start_y_m
                + fraction * delta_y_m
            ),
            vx_mps=vx_mps,
            vy_mps=vy_mps,
        )


# ============================================================
# Motion wrappers
# ============================================================

def move_forward(
    mc: MotionCommander,
    distance_m: float,
    label: str,
) -> None:
    global pose

    print(label)

    start_x = pose.x
    start_y = pose.y
    command_start = elapsed_s()

    mc.forward(
        distance_m,
        velocity=FORWARD_SPEED_MPS,
    )

    record_linear_segment(
        start_x_m=start_x,
        start_y_m=start_y,
        delta_x_m=distance_m,
        delta_y_m=0.0,
        speed_mps=FORWARD_SPEED_MPS,
        command_start_s=command_start,
        label=label,
        vx_mps=FORWARD_SPEED_MPS,
        vy_mps=0.0,
    )

    pose.x += distance_m

    append_log(
        timestamp_s=elapsed_s(),
        label=label,
        event="COMMAND_COMPLETE",
        x_m=pose.x,
        y_m=pose.y,
        vx_mps=0.0,
        vy_mps=0.0,
    )

    time.sleep(COMMAND_PAUSE_S)


def move_right(
    mc: MotionCommander,
    distance_m: float,
    label: str,
) -> None:
    global pose

    print(label)

    start_x = pose.x
    start_y = pose.y
    command_start = elapsed_s()

    mc.right(
        distance_m,
        velocity=LATERAL_SPEED_MPS,
    )

    record_linear_segment(
        start_x_m=start_x,
        start_y_m=start_y,
        delta_x_m=0.0,
        delta_y_m=-distance_m,
        speed_mps=LATERAL_SPEED_MPS,
        command_start_s=command_start,
        label=label,
        vx_mps=0.0,
        vy_mps=-LATERAL_SPEED_MPS,
    )

    pose.y -= distance_m

    append_log(
        timestamp_s=elapsed_s(),
        label=label,
        event="COMMAND_COMPLETE",
        x_m=pose.x,
        y_m=pose.y,
        vx_mps=0.0,
        vy_mps=0.0,
    )

    time.sleep(COMMAND_PAUSE_S)


def move_left(
    mc: MotionCommander,
    distance_m: float,
    label: str,
) -> None:
    global pose

    print(label)

    start_x = pose.x
    start_y = pose.y
    command_start = elapsed_s()

    mc.left(
        distance_m,
        velocity=LATERAL_SPEED_MPS,
    )

    record_linear_segment(
        start_x_m=start_x,
        start_y_m=start_y,
        delta_x_m=0.0,
        delta_y_m=distance_m,
        speed_mps=LATERAL_SPEED_MPS,
        command_start_s=command_start,
        label=label,
        vx_mps=0.0,
        vy_mps=LATERAL_SPEED_MPS,
    )

    pose.y += distance_m

    append_log(
        timestamp_s=elapsed_s(),
        label=label,
        event="COMMAND_COMPLETE",
        x_m=pose.x,
        y_m=pose.y,
        vx_mps=0.0,
        vy_mps=0.0,
    )

    time.sleep(COMMAND_PAUSE_S)


# ============================================================
# Save CSV and paper figures
# ============================================================

def save_results() -> None:
    RUN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(logs)

    dataframe.to_csv(
        CSV_PATH,
        index=False,
    )

    # Trajectory
    figure, axis = plt.subplots(
        figsize=(8, 4.5)
    )

    axis.plot(
        dataframe["x_est_m"],
        dataframe["y_est_m"],
        linewidth=2.0,
        label="Command-derived path",
    )

    axis.scatter(
        [START_X_M],
        [START_Y_M],
        s=70,
        label="Takeoff",
    )

    axis.scatter(
        [pose.x],
        [pose.y],
        marker="*",
        s=130,
        label="Landing",
    )

    for obstacle in OBSTACLES:
        obstacle_circle = plt.Circle(
            (
                obstacle["cx"],
                obstacle["cy"],
            ),
            obstacle["radius"],
            fill=False,
            linestyle="--",
            linewidth=2,
        )

        safe_circle = plt.Circle(
            (
                obstacle["cx"],
                obstacle["cy"],
            ),
            (
                obstacle["radius"]
                + SAFE_CLEARANCE_M
            ),
            fill=False,
            linestyle=":",
            linewidth=1,
        )

        axis.add_patch(
            obstacle_circle
        )
        axis.add_patch(
            safe_circle
        )

        axis.text(
            obstacle["cx"],
            obstacle["cy"],
            obstacle["name"],
            ha="center",
            va="center",
        )

    monitor_rows = dataframe[
        dataframe["monitor_active"] == 1
    ]

    if not monitor_rows.empty:
        axis.scatter(
            monitor_rows["x_est_m"],
            monitor_rows["y_est_m"],
            marker="x",
            s=25,
            label="Monitor active",
        )

    axis.set_xlim(
        0.0,
        MAP_LENGTH_X_M,
    )
    axis.set_ylim(
        0.0,
        MAP_WIDTH_Y_M,
    )
    axis.set_aspect(
        "equal",
        adjustable="box",
    )
    axis.set_xlabel(
        "command-derived x [m]"
    )
    axis.set_ylabel(
        "command-derived y [m]"
    )
    axis.set_title(
        "Crazyflie straight-motion two-obstacle route"
    )
    axis.grid(True)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(
        TRAJECTORY_PATH,
        dpi=300,
    )
    plt.close(figure)

    # Clearance
    figure, axis = plt.subplots(
        figsize=(8, 4)
    )

    axis.plot(
        dataframe["time_s"],
        dataframe["clearance_m"],
        linewidth=1.8,
        label="Estimated clearance",
    )

    axis.axhline(
        SAFE_CLEARANCE_M,
        linestyle="--",
        label="Safe threshold",
    )

    axis.axhline(
        WARNING_CLEARANCE_M,
        linestyle=":",
        label="Warning threshold",
    )

    axis.axhline(
        0.0,
        linewidth=0.8,
        label="Collision boundary",
    )

    axis.set_xlabel("time [s]")
    axis.set_ylabel("clearance [m]")
    axis.set_title(
        "Nearest-obstacle clearance"
    )
    axis.grid(True)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(
        CLEARANCE_PATH,
        dpi=300,
    )
    plt.close(figure)

    # Risk
    figure, axis = plt.subplots(
        figsize=(8, 4)
    )

    axis.plot(
        dataframe["time_s"],
        dataframe["risk"],
        linewidth=1.8,
        label="Risk score",
    )

    axis.set_xlabel("time [s]")
    axis.set_ylabel("risk")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(
        "Fixed-map risk response"
    )
    axis.grid(True)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(
        RISK_PATH,
        dpi=300,
    )
    plt.close(figure)

    # Commands
    figure, axis = plt.subplots(
        figsize=(8, 4)
    )

    axis.plot(
        dataframe["time_s"],
        dataframe["vx_mps"],
        linewidth=1.6,
        label="$v_x$",
    )

    axis.plot(
        dataframe["time_s"],
        dataframe["vy_mps"],
        linewidth=1.6,
        label="$v_y$",
    )

    axis.set_xlabel("time [s]")
    axis.set_ylabel(
        "velocity command [m/s]"
    )
    axis.set_title(
        "Forward and lateral motion commands"
    )
    axis.grid(True)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(
        COMMANDS_PATH,
        dpi=300,
    )
    plt.close(figure)

    print(f"CSV saved: {CSV_PATH}")
    print(
        f"Trajectory saved: "
        f"{TRAJECTORY_PATH}"
    )
    print(
        f"Clearance saved: "
        f"{CLEARANCE_PATH}"
    )
    print(f"Risk saved: {RISK_PATH}")
    print(
        f"Commands saved: "
        f"{COMMANDS_PATH}"
    )

    print("\nExperiment summary")
    print(
        "Minimum command-derived clearance [m]:",
        round(
            float(
                dataframe[
                    "clearance_m"
                ].min()
            ),
            3,
        ),
    )
    print(
        "Maximum risk:",
        round(
            float(
                dataframe["risk"].max()
            ),
            3,
        ),
    )
    print(
        "Landing estimate:",
        (
            round(pose.x, 3),
            round(pose.y, 3),
        ),
    )


# ============================================================
# Safe shutdown
# ============================================================

def safe_land_and_disarm(
    scf: SyncCrazyflie,
    mc: MotionCommander | None,
    already_landed: bool,
) -> None:
    print("Stopping safely...")

    if (
        mc is not None
        and not already_landed
    ):
        try:
            mc.stop()
            time.sleep(0.3)
        except Exception:
            pass

        try:
            mc.land(
                velocity=VERTICAL_SPEED_MPS
            )
            time.sleep(1.5)
        except Exception:
            pass

    try:
        scf.cf.commander.send_stop_setpoint()
        time.sleep(0.2)
    except Exception:
        pass

    try:
        scf.cf.platform.send_arming_request(
            False
        )
        time.sleep(0.5)
    except Exception:
        pass

    print("Crazyflie disarmed.")


# ============================================================
# Main flight
# ============================================================

def main() -> None:
    global experiment_start_s
    global pose

    validate_route()

    RUN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pose = Pose2D(
        x=START_X_M,
        y=START_Y_M,
    )

    logs.clear()

    cflib.crtp.init_drivers(
        enable_debug_driver=False
    )

    mc: MotionCommander | None = None
    already_landed = False

    print("Connecting to Crazyflie...")

    with SyncCrazyflie(URI) as scf:
        print("Crazyflie connected.")

        try:
            print("Arming...")

            scf.cf.platform.send_arming_request(
                True
            )

            time.sleep(1.0)

            mc = MotionCommander(
                scf,
                default_height=FLIGHT_HEIGHT_M,
            )

            experiment_start_s = (
                time.monotonic()
            )

            append_log(
                timestamp_s=0.0,
                label="takeoff_start",
                event="TAKEOFF",
                x_m=pose.x,
                y_m=pose.y,
                vx_mps=0.0,
                vy_mps=0.0,
            )

            print(
                f"Taking off to "
                f"{FLIGHT_HEIGHT_M:.2f} m..."
            )

            mc.take_off(
                height=FLIGHT_HEIGHT_M,
                velocity=VERTICAL_SPEED_MPS,
            )

            time.sleep(1.5)

            append_log(
                timestamp_s=elapsed_s(),
                label="takeoff_complete",
                event="HOVER",
                x_m=pose.x,
                y_m=pose.y,
                vx_mps=0.0,
                vy_mps=0.0,
            )

            # ================================================
            # Straight-motion sequence
            # ================================================

            move_forward(
                mc,
                APPROACH_DISTANCE_M,
                "1. Forward 12 cm",
            )

            move_right(
                mc,
                RIGHT_SHIFT_M,
                "2. Right 16 cm",
            )

            move_forward(
                mc,
                PASS_DISTANCE_M,
                "3. Forward 62 cm past both obstacles",
            )

            move_left(
                mc,
                LEFT_SHIFT_M,
                "4. Left 16 cm",
            )

            move_forward(
                mc,
                FINAL_FORWARD_M,
                "5. Forward 10 cm to landing point",
            )

            print(
                "Both obstacles passed. Landing..."
            )

            append_log(
                timestamp_s=elapsed_s(),
                label="landing_start",
                event="LANDING",
                x_m=pose.x,
                y_m=pose.y,
                vx_mps=0.0,
                vy_mps=0.0,
            )

            mc.land(
                velocity=VERTICAL_SPEED_MPS
            )

            time.sleep(1.5)

            already_landed = True

            append_log(
                timestamp_s=elapsed_s(),
                label="landing_complete",
                event="LANDED",
                x_m=pose.x,
                y_m=pose.y,
                vx_mps=0.0,
                vy_mps=0.0,
            )

        except KeyboardInterrupt:
            print("Ctrl+C pressed.")

        finally:
            safe_land_and_disarm(
                scf=scf,
                mc=mc,
                already_landed=already_landed,
            )

    save_results()


if __name__ == "__main__":
    main()