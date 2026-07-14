"""
Create clear paper-oriented diagnostics from the oracle comparison results.

Run from the repository root:

python -m gym_pybullet_drones.predictiveMonitoringControl.plot_paper_diagnostics \
    --comparison_dir results/oracle_comparison/20260711_144257

You may also point it to the parent folder. In that case, the newest timestamped
comparison directory is selected automatically:

python -m gym_pybullet_drones.predictiveMonitoringControl.plot_paper_diagnostics \
    --comparison_dir results/oracle_comparison

Outputs:
- per_obstacle_summary.csv
- clearance_vs_x.png / .pdf
- risk_vs_estimated_distance.png / .pdf
- risk_vs_true_clearance_by_obstacle.png / .pdf
- closed_loop_obstacle_response.png / .pdf
- obstacle_O1_comparison.png / .pdf
- obstacle_O2_comparison.png / .pdf
- obstacle_O3_comparison.png / .pdf
- obstacle_O4_comparison.png / .pdf
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


START = np.array([-1.20, 0.00], dtype=float)
GOAL = np.array([4.20, 0.00], dtype=float)
SAFETY_MARGIN = 0.18

WARNING_DISTANCE_M = 1.20
CRITICAL_DISTANCE_M = 0.65


@dataclass(frozen=True)
class Obstacle:
    name: str
    camera_name: str
    cx: float
    cy: float
    radius: float


OBSTACLES = (
    Obstacle("O1", "O1_RED", -0.10, 0.18, 0.20),
    Obstacle("O2", "O2_BLUE", 1.15, -0.18, 0.20),
    Obstacle("O3", "O3_YELLOW", 2.40, 0.18, 0.20),
    Obstacle("O4", "O4_GREEN", 3.65, -0.18, 0.20),
)


METHOD_LABELS = {
    "geometry": "Geometry controller",
    "geometry_perception": "Geometry + perception monitoring",
    "geometry_perception_state": (
        "Geometry + perception/state monitoring"
    ),
    "closed_loop_camera": "Closed-loop camera oracle",
}


def select_comparison_directory(path: Path) -> Path:
    if (path / "trial_metrics.csv").exists():
        return path

    candidates = [
        item
        for item in path.iterdir()
        if item.is_dir()
        and (item / "trial_metrics.csv").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            "No comparison directory containing trial_metrics.csv "
            f"was found under: {path}"
        )

    return max(
        candidates,
        key=lambda item: item.stat().st_mtime,
    )


def find_trial_csv(
    comparison_dir: Path,
    method_key: str,
    trial_index: int,
) -> Path:
    trial_folder = (
        comparison_dir
        / method_key
        / f"trial_{trial_index:03d}"
    )

    candidates = list(
        trial_folder.rglob("*trajectory*.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No trajectory CSV found for {method_key}, "
            f"trial {trial_index}, under {trial_folder}"
        )

    return max(
        candidates,
        key=lambda item: item.stat().st_mtime,
    )


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def numeric_series(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
    default: float = np.nan,
) -> pd.Series:
    column = first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        return pd.Series(
            np.full(len(dataframe), default),
            index=dataframe.index,
            dtype=float,
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def text_series(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
    default: str = "",
) -> pd.Series:
    column = first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        return pd.Series(
            [default] * len(dataframe),
            index=dataframe.index,
            dtype=str,
        )

    return dataframe[column].fillna(default).astype(str)


def standardize_dataframe(
    csv_path: Path,
    method_key: str,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)

    dataframe = pd.DataFrame(
        {
            "time_s": numeric_series(
                raw,
                ("time_s", "time"),
            ),
            "x_m": numeric_series(
                raw,
                ("x_m", "x"),
            ),
            "y_m": numeric_series(
                raw,
                ("y_m", "y"),
            ),
            "physical_clearance_m": numeric_series(
                raw,
                (
                    "physical_clearance_m",
                    "physical_clearance",
                    "current_clearance_m",
                ),
            ),
            "effective_clearance_m": numeric_series(
                raw,
                (
                    "effective_clearance_m",
                    "effective_clearance",
                ),
            ),
            "speed_mps": numeric_series(
                raw,
                (
                    "commanded_speed_mps",
                    "executed_speed_mps",
                    "speed",
                ),
            ),
            "risk_score": numeric_series(
                raw,
                (
                    "risk_score",
                    "proximity_risk",
                    "risk",
                ),
            ),
            "estimated_distance_m": numeric_series(
                raw,
                (
                    "nearest_detected_distance_m",
                    "estimated_distance_m",
                    "camera_distance_m",
                ),
            ),
            "confidence": numeric_series(
                raw,
                (
                    "nearest_detected_confidence",
                    "confidence",
                ),
            ),
            "control_mode": text_series(
                raw,
                ("control_mode", "mode"),
                default="UNKNOWN",
            ),
            "nearest_true_obstacle": text_series(
                raw,
                (
                    "nearest_true_obstacle",
                    "nearest_obstacle",
                ),
                default="",
            ),
            "active_camera_obstacle": text_series(
                raw,
                (
                    "active_camera_obstacle",
                    "active_obstacle",
                ),
                default="",
            ),
        }
    )

    dataframe["method_key"] = method_key
    dataframe["method"] = METHOD_LABELS.get(
        method_key,
        method_key,
    )

    dataframe = dataframe.dropna(
        subset=["time_s", "x_m", "y_m"]
    ).reset_index(drop=True)

    return dataframe


def add_per_obstacle_clearances(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    for obstacle in OBSTACLES:
        centre_distance = np.sqrt(
            (dataframe["x_m"] - obstacle.cx) ** 2
            + (dataframe["y_m"] - obstacle.cy) ** 2
        )

        dataframe[
            f"{obstacle.name}_clearance_m"
        ] = centre_distance - obstacle.radius

    return dataframe


def closest_event(
    dataframe: pd.DataFrame,
    obstacle: Obstacle,
) -> dict[str, float | str | int]:
    clearance_column = (
        f"{obstacle.name}_clearance_m"
    )

    index = dataframe[
        clearance_column
    ].idxmin()

    row = dataframe.loc[index]

    return {
        "method_key": str(row["method_key"]),
        "method": str(row["method"]),
        "obstacle": obstacle.name,
        "camera_obstacle": obstacle.camera_name,
        "closest_index": int(index),
        "closest_time_s": float(row["time_s"]),
        "closest_x_m": float(row["x_m"]),
        "closest_y_m": float(row["y_m"]),
        "minimum_clearance_m": float(
            row[clearance_column]
        ),
        "effective_clearance_m": float(
            row[clearance_column]
            - SAFETY_MARGIN
        ),
        "risk_at_closest": float(
            row["risk_score"]
        )
        if np.isfinite(row["risk_score"])
        else np.nan,
        "speed_at_closest_mps": float(
            row["speed_mps"]
        )
        if np.isfinite(row["speed_mps"])
        else np.nan,
        "estimated_distance_at_closest_m": float(
            row["estimated_distance_m"]
        )
        if np.isfinite(row["estimated_distance_m"])
        else np.nan,
        "control_mode_at_closest": str(
            row["control_mode"]
        ),
    }


def local_window(
    dataframe: pd.DataFrame,
    event: dict,
    window_s: float,
) -> pd.DataFrame:
    time_zero = float(
        event["closest_time_s"]
    )

    local = dataframe[
        (
            dataframe["time_s"]
            >= time_zero - window_s
        )
        & (
            dataframe["time_s"]
            <= time_zero + window_s
        )
    ].copy()

    local["relative_time_s"] = (
        local["time_s"] - time_zero
    )

    return local


def active_window_for_obstacle(
    dataframe: pd.DataFrame,
    obstacle: Obstacle,
    event: dict,
    fallback_window_s: float,
) -> pd.DataFrame:
    active_names = dataframe[
        "active_camera_obstacle"
    ].astype(str)

    mask = active_names.str.contains(
        obstacle.camera_name,
        regex=False,
    )

    active = dataframe[mask].copy()

    if len(active) < 2:
        active = local_window(
            dataframe,
            event,
            fallback_window_s,
        )
    else:
        active["relative_time_s"] = (
            active["time_s"]
            - float(event["closest_time_s"])
        )

    return active


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    base_name: str,
) -> None:
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_clearance_vs_x(
    dataframes: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10.5, 5.2)
    )

    for method_key, dataframe in dataframes.items():
        axis.plot(
            dataframe["x_m"],
            dataframe["physical_clearance_m"],
            linewidth=2.0,
            label=METHOD_LABELS.get(
                method_key,
                method_key,
            ),
        )

    for obstacle in OBSTACLES:
        axis.axvline(
            obstacle.cx,
            linestyle=":",
            linewidth=1.0,
        )
        axis.text(
            obstacle.cx,
            axis.get_ylim()[1]
            if axis.get_ylim()[1] > 0
            else 1.0,
            obstacle.name,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Collision boundary",
    )
    axis.axhline(
        SAFETY_MARGIN,
        linestyle=":",
        linewidth=1.4,
        label="Safety-margin boundary",
    )

    axis.set_title(
        "Physical clearance along the flight path"
    )
    axis.set_xlabel("drone x position [m]")
    axis.set_ylabel(
        "nearest physical clearance [m]"
    )
    axis.grid(True)
    axis.legend(fontsize=8)
    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "clearance_vs_x",
    )


def plot_risk_vs_estimated_distance(
    closed_loop: pd.DataFrame,
    output_dir: Path,
) -> None:
    valid = closed_loop[
        np.isfinite(
            closed_loop["estimated_distance_m"]
        )
        & np.isfinite(
            closed_loop["risk_score"]
        )
    ].copy()

    if valid.empty:
        print(
            "No valid estimated-distance/risk samples; "
            "risk_vs_estimated_distance skipped."
        )
        return

    distance_values = np.linspace(
        max(
            0.0,
            float(
                valid[
                    "estimated_distance_m"
                ].min()
            ),
        ),
        float(
            valid[
                "estimated_distance_m"
            ].max()
        ),
        300,
    )

    denominator = (
        WARNING_DISTANCE_M
        - CRITICAL_DISTANCE_M
    )

    theoretical_risk = np.clip(
        (
            WARNING_DISTANCE_M
            - distance_values
        )
        / denominator,
        0.0,
        1.0,
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 5.2)
    )

    axis.scatter(
        valid["estimated_distance_m"],
        valid["risk_score"],
        s=12,
        alpha=0.30,
        label="Recorded camera-oracle samples",
    )

    axis.plot(
        distance_values,
        theoretical_risk,
        linewidth=2.4,
        label="Configured risk policy",
    )

    axis.axvline(
        WARNING_DISTANCE_M,
        linestyle="--",
        linewidth=1.2,
        label="Warning distance",
    )

    axis.axvline(
        CRITICAL_DISTANCE_M,
        linestyle=":",
        linewidth=1.4,
        label="Critical distance",
    )

    axis.set_title(
        "Oracle risk versus camera-estimated distance"
    )
    axis.set_xlabel(
        "camera-estimated obstacle distance [m]"
    )
    axis.set_ylabel("oracle risk score")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True)
    axis.legend(fontsize=8)
    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "risk_vs_estimated_distance",
    )


def plot_risk_vs_true_clearance_by_obstacle(
    closed_loop: pd.DataFrame,
    events: dict[str, dict],
    output_dir: Path,
    window_s: float,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.5),
    )

    for axis, obstacle in zip(
        axes.ravel(),
        OBSTACLES,
    ):
        event = events[obstacle.name]

        active = active_window_for_obstacle(
            closed_loop,
            obstacle,
            event,
            window_s,
        )

        clearance_column = (
            f"{obstacle.name}_clearance_m"
        )

        valid = active[
            np.isfinite(active[clearance_column])
            & np.isfinite(
                active["risk_score"]
            )
        ].copy()

        if valid.empty:
            axis.set_title(
                f"{obstacle.name}: no valid risk data"
            )
            axis.grid(True)
            continue

        approaching = valid[
            valid["relative_time_s"] <= 0.0
        ]
        departing = valid[
            valid["relative_time_s"] > 0.0
        ]

        axis.scatter(
            approaching[clearance_column],
            approaching["risk_score"],
            s=14,
            alpha=0.45,
            label="Approach",
        )

        axis.scatter(
            departing[clearance_column],
            departing["risk_score"],
            s=14,
            alpha=0.45,
            label="Departure",
        )

        axis.axvline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            label="Collision",
        )

        axis.axvline(
            SAFETY_MARGIN,
            linestyle=":",
            linewidth=1.2,
            label="Safety margin",
        )

        axis.set_title(
            f"{obstacle.name}: risk vs true clearance"
        )
        axis.set_xlabel("true clearance [m]")
        axis.set_ylabel("risk")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True)
        axis.legend(fontsize=7)

    figure.suptitle(
        "Camera-oracle risk versus true clearance by obstacle",
        fontsize=13,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        figure,
        output_dir,
        "risk_vs_true_clearance_by_obstacle",
    )


def plot_closed_loop_obstacle_response(
    closed_loop: pd.DataFrame,
    events: dict[str, dict],
    output_dir: Path,
    window_s: float,
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
    )

    for obstacle in OBSTACLES:
        event = events[obstacle.name]

        local = local_window(
            closed_loop,
            event,
            window_s,
        )

        clearance_column = (
            f"{obstacle.name}_clearance_m"
        )

        axes[0].plot(
            local["relative_time_s"],
            local[clearance_column],
            linewidth=2.0,
            label=obstacle.name,
        )

        axes[1].plot(
            local["relative_time_s"],
            local["risk_score"],
            linewidth=2.0,
            label=obstacle.name,
        )

        axes[2].plot(
            local["relative_time_s"],
            local["speed_mps"],
            linewidth=2.0,
            label=obstacle.name,
        )

    axes[0].axhline(
        0.0,
        linestyle="--",
        linewidth=1.1,
        label="Collision",
    )
    axes[0].axhline(
        SAFETY_MARGIN,
        linestyle=":",
        linewidth=1.2,
        label="Safety margin",
    )

    axes[0].set_ylabel("clearance [m]")
    axes[0].set_title(
        "Clearance around each closest pass"
    )
    axes[0].grid(True)
    axes[0].legend(
        fontsize=8,
        ncol=3,
    )

    axes[1].set_ylabel("risk")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title(
        "Risk response around each obstacle"
    )
    axes[1].grid(True)
    axes[1].legend(
        fontsize=8,
        ncol=4,
    )

    axes[2].set_ylabel("commanded speed [m/s]")
    axes[2].set_xlabel(
        "time relative to closest pass [s]"
    )
    axes[2].set_title(
        "Speed response around each obstacle"
    )
    axes[2].grid(True)
    axes[2].legend(
        fontsize=8,
        ncol=4,
    )

    for axis in axes:
        axis.axvline(
            0.0,
            linestyle=":",
            linewidth=1.1,
        )

    figure.suptitle(
        "Closed-loop response when each obstacle is encountered",
        fontsize=13,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        figure,
        output_dir,
        "closed_loop_obstacle_response",
    )


def plot_single_obstacle_comparison(
    obstacle: Obstacle,
    dataframes: dict[str, pd.DataFrame],
    events_by_method: dict[str, dict[str, dict]],
    output_dir: Path,
    window_s: float,
) -> None:
    figure = plt.figure(
        figsize=(11.0, 7.2)
    )

    grid = figure.add_gridspec(
        3,
        2,
    )

    path_axis = figure.add_subplot(
        grid[:, 0]
    )
    clearance_axis = figure.add_subplot(
        grid[0, 1]
    )
    risk_axis = figure.add_subplot(
        grid[1, 1]
    )
    speed_axis = figure.add_subplot(
        grid[2, 1]
    )

    for method_key, dataframe in dataframes.items():
        event = events_by_method[
            method_key
        ][obstacle.name]

        local = local_window(
            dataframe,
            event,
            window_s,
        )

        label = METHOD_LABELS.get(
            method_key,
            method_key,
        )

        # Use a spatial window for the path plot so every method is shown
        # across the same obstacle neighbourhood. A time window can truncate
        # slower or faster trajectories differently.
        spatial_margin = 0.75
        spatial = dataframe[
            (dataframe["x_m"] >= obstacle.cx - spatial_margin)
            & (dataframe["x_m"] <= obstacle.cx + spatial_margin)
            & (dataframe["y_m"] >= obstacle.cy - spatial_margin)
            & (dataframe["y_m"] <= obstacle.cy + spatial_margin)
        ].copy()

        path_axis.plot(
            spatial["x_m"],
            spatial["y_m"],
            linewidth=2.0,
            label=label,
        )

        clearance_axis.plot(
            local["relative_time_s"],
            local[
                f"{obstacle.name}_clearance_m"
            ],
            linewidth=1.8,
            label=label,
        )

        speed_axis.plot(
            local["relative_time_s"],
            local["speed_mps"],
            linewidth=1.8,
            label=label,
        )

        if np.isfinite(
            local["risk_score"]
        ).any():
            risk_axis.plot(
                local["relative_time_s"],
                local["risk_score"],
                linewidth=2.0,
                label=label,
            )

    path_axis.add_patch(
        plt.Circle(
            (obstacle.cx, obstacle.cy),
            obstacle.radius,
            fill=False,
            linewidth=2.0,
        )
    )

    path_axis.add_patch(
        plt.Circle(
            (obstacle.cx, obstacle.cy),
            obstacle.radius
            + SAFETY_MARGIN,
            fill=False,
            linestyle="--",
            linewidth=1.5,
        )
    )

    path_axis.scatter(
        [obstacle.cx],
        [obstacle.cy],
        marker="x",
        s=60,
    )

    path_axis.text(
        obstacle.cx,
        obstacle.cy,
        obstacle.name,
        ha="center",
        va="center",
        fontweight="bold",
    )

    margin = 0.75

    path_axis.set_xlim(
        obstacle.cx - margin,
        obstacle.cx + margin,
    )
    path_axis.set_ylim(
        obstacle.cy - margin,
        obstacle.cy + margin,
    )
    path_axis.set_aspect(
        "equal",
        adjustable="box",
    )
    path_axis.set_xlabel("x position [m]")
    path_axis.set_ylabel("y position [m]")
    path_axis.set_title(
        f"{obstacle.name}: local paths"
    )
    path_axis.grid(True)
    path_axis.legend(
        fontsize=7,
        loc="best",
    )

    clearance_axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )
    clearance_axis.axhline(
        SAFETY_MARGIN,
        linestyle=":",
        linewidth=1.1,
    )
    clearance_axis.axvline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )
    clearance_axis.set_ylabel("clearance [m]")
    clearance_axis.set_title(
        "Clearance near closest pass"
    )
    clearance_axis.grid(True)

    risk_axis.axvline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )
    risk_axis.set_ylabel("risk")
    risk_axis.set_ylim(-0.03, 1.03)
    risk_axis.set_title(
        "Camera-oracle risk"
    )
    risk_axis.grid(True)

    speed_axis.axvline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )
    speed_axis.set_ylabel("speed [m/s]")
    speed_axis.set_xlabel(
        "time relative to closest pass [s]"
    )
    speed_axis.set_title(
        "Commanded speed near obstacle"
    )
    speed_axis.grid(True)
    speed_axis.legend(
        fontsize=7,
        loc="best",
    )

    figure.suptitle(
        f"{obstacle.name} comparison: "
        "path, clearance, risk, and speed",
        fontsize=13,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        figure,
        output_dir,
        f"obstacle_{obstacle.name}_comparison",
    )


def build_summary(
    dataframes: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, dict]],
]:
    records: list[dict] = []
    events_by_method: dict[
        str,
        dict[str, dict],
    ] = {}

    for method_key, dataframe in dataframes.items():
        events_by_method[
            method_key
        ] = {}

        for obstacle in OBSTACLES:
            event = closest_event(
                dataframe,
                obstacle,
            )

            events_by_method[
                method_key
            ][obstacle.name] = event

            local = local_window(
                dataframe,
                event,
                window_s=8.0,
            )

            event[
                "maximum_local_risk"
            ] = (
                float(
                    np.nanmax(
                        local["risk_score"]
                    )
                )
                if np.isfinite(
                    local["risk_score"]
                ).any()
                else np.nan
            )

            event[
                "minimum_local_speed_mps"
            ] = (
                float(
                    np.nanmin(
                        local["speed_mps"]
                    )
                )
                if np.isfinite(
                    local["speed_mps"]
                ).any()
                else np.nan
            )

            records.append(event)

    return (
        pd.DataFrame(records),
        events_by_method,
    )


def print_closed_loop_summary(
    summary: pd.DataFrame,
) -> None:
    closed = summary[
        summary["method_key"]
        == "closed_loop_camera"
    ].copy()

    columns = [
        "obstacle",
        "closest_time_s",
        "minimum_clearance_m",
        "risk_at_closest",
        "maximum_local_risk",
        "speed_at_closest_mps",
        "minimum_local_speed_mps",
        "control_mode_at_closest",
    ]

    print(
        "\nClosed-loop per-obstacle response:"
    )
    print(
        closed[columns].to_string(
            index=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate understandable paper figures and "
            "per-obstacle diagnostics from comparison results."
        )
    )

    parser.add_argument(
        "--comparison_dir",
        required=True,
        help=(
            "Timestamped comparison directory, or its "
            "parent results/oracle_comparison directory."
        ),
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help=(
            "Trial number used for trajectory figures."
        ),
    )

    parser.add_argument(
        "--window",
        type=float,
        default=12.0,
        help=(
            "Seconds before and after closest passage "
            "shown in local obstacle plots."
        ),
    )

    parser.add_argument(
        "--out_dir",
        default=None,
        help=(
            "Optional output directory. Default: "
            "<comparison_dir>/paper_diagnostics"
        ),
    )

    arguments = parser.parse_args()

    comparison_dir = select_comparison_directory(
        Path(arguments.comparison_dir)
    )

    output_dir = (
        Path(arguments.out_dir)
        if arguments.out_dir
        else comparison_dir
        / "paper_diagnostics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trial_metrics = pd.read_csv(
        comparison_dir
        / "trial_metrics.csv"
    )

    available_method_keys = list(
        trial_metrics[
            "method_key"
        ].dropna().astype(str).unique()
    )

    dataframes: dict[
        str,
        pd.DataFrame,
    ] = {}

    for method_key in available_method_keys:
        csv_path = find_trial_csv(
            comparison_dir,
            method_key,
            arguments.trial,
        )

        dataframe = standardize_dataframe(
            csv_path,
            method_key,
        )

        dataframe = add_per_obstacle_clearances(
            dataframe
        )

        dataframes[
            method_key
        ] = dataframe

        print(
            f"Loaded {METHOD_LABELS.get(method_key, method_key)}: "
            f"{csv_path}"
        )

    summary, events_by_method = build_summary(
        dataframes
    )

    summary_path = (
        output_dir
        / "per_obstacle_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"Saved per-obstacle summary: "
        f"{summary_path}"
    )

    plot_clearance_vs_x(
        dataframes,
        output_dir,
    )

    closed_loop = dataframes.get(
        "closed_loop_camera"
    )

    if closed_loop is not None:
        closed_events = events_by_method[
            "closed_loop_camera"
        ]

        plot_risk_vs_estimated_distance(
            closed_loop,
            output_dir,
        )

        plot_risk_vs_true_clearance_by_obstacle(
            closed_loop,
            closed_events,
            output_dir,
            arguments.window,
        )

        plot_closed_loop_obstacle_response(
            closed_loop,
            closed_events,
            output_dir,
            arguments.window,
        )

    for obstacle in OBSTACLES:
        plot_single_obstacle_comparison(
            obstacle,
            dataframes,
            events_by_method,
            output_dir,
            arguments.window,
        )

    print_closed_loop_summary(
        summary
    )

    print(
        f"\nAll paper diagnostics saved in: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()