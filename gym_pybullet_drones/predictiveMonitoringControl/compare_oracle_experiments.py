"""
Run and compare the four UAV experiment variants.

Run from the repository root:

python -m gym_pybullet_drones.predictiveMonitoringControl.compare_oracle_experiments \
    --trials 1 --duration_sec 180

Outputs are written under:
results/oracle_comparison/<timestamp>/

The script runs:
1. Geometry oracle
2. Geometry oracle + perception
3. Geometry oracle + perception + state estimation
4. Closed-loop camera oracle

It then produces:
- trial_metrics.csv
- method_summary.csv
- trajectory_comparison.png/pdf
- clearance_over_time.png/pdf
- risk_vs_clearance.png/pdf
- intervention_fraction.png/pdf
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_PREFIX = "gym_pybullet_drones.predictiveMonitoringControl"

START = np.array([-1.20, 0.00], dtype=float)
GOAL = np.array([4.20, 0.00], dtype=float)
GOAL_TOLERANCE = 0.12


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    module: str

'''
METHODS = (
    MethodSpec(
        key="geometry",
        label="Geometry oracle",
        module=f"{PACKAGE_PREFIX}.run_geometry_oracle",
    ),
    MethodSpec(
        key="geometry_perception",
        label="Geometry + perception",
        module=f"{PACKAGE_PREFIX}.run_geometry_oracle_with_perception",
    ),
    
    MethodSpec(
        key="geometry_perception_state",
        label="Geometry + perception + state estimation",
        module=(
            f"{PACKAGE_PREFIX}."
            "run_camera_oracle_with_perception_stateEstimation"
        ),
    ),
    MethodSpec(
        key="closed_loop_camera",
        label="Closed-loop camera oracle",
        module=f"{PACKAGE_PREFIX}.run_closed_loop_camera_oracle",
    ),
)
'''
'''
METHODS = (
    MethodSpec(
        key="geometry",
        label="Ideal geometry oracle",
        module=f"{PACKAGE_PREFIX}.run_geometry_oracle",
    ),
    MethodSpec(
        key="geometry_perception",
        label="Perception without state estimation",
        module=f"{PACKAGE_PREFIX}.run_geometry_oracle_with_perception",
    ),
    MethodSpec(
        key="closed_loop_camera",
        label="Proposed closed-loop camera oracle",
        module=f"{PACKAGE_PREFIX}.run_closed_loop_camera_oracle",
    ),
)
'''

METHODS = (
    MethodSpec(
        key="geometry",
        label="Ideal geometry oracle",
        module=f"{PACKAGE_PREFIX}.run_geometry_oracle",
    ),
    MethodSpec(
        key="closed_loop_camera",
        label="Proposed closed-loop camera oracle",
        module=f"{PACKAGE_PREFIX}.run_closed_loop_camera_oracle",
    ),
)


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def first_existing(
    row: dict[str, str],
    candidates: Iterable[str],
    default: str = "",
) -> str:
    for key in candidates:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def find_trajectory_csv(output_folder: Path) -> Path:
    candidates = list(
        output_folder.rglob("*trajectory*.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No trajectory CSV was created under {output_folder}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def run_method(
    method: MethodSpec,
    trial_index: int,
    duration_sec: int,
    comparison_root: Path,
    python_executable: str,
    verbose: bool,
) -> Path:
    method_output = (
        comparison_root
        / method.key
        / f"trial_{trial_index:03d}"
    )
    method_output.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable,
        "-m",
        method.module,
        "--gui",
        "False",
        "--plot",
        "False",
        "--duration_sec",
        str(duration_sec),
        "--output_folder",
        str(method_output),
    ]

    print(
        f"\n[{method.label}] Trial {trial_index}: "
        + " ".join(command)
    )

    start = time.perf_counter()

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    elapsed = time.perf_counter() - start

    log_path = method_output / "process_output.txt"
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )

    if verbose:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"{method.label}, trial {trial_index}, failed with "
            f"return code {completed.returncode}. "
            f"See {log_path}"
        )

    csv_path = find_trajectory_csv(method_output)

    print(
        f"[{method.label}] completed in {elapsed:.1f} s; "
        f"trajectory: {csv_path}"
    )

    return csv_path


def calculate_path_length(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    if len(x_values) < 2:
        return 0.0

    dx = np.diff(x_values)
    dy = np.diff(y_values)

    return float(
        np.sum(np.sqrt(dx * dx + dy * dy))
    )


def count_intervention_events(modes: list[str]) -> int:
    event_count = 0
    previous_is_intervention = False

    for mode in modes:
        upper_mode = mode.upper()

        is_intervention = (
            "AVOID" in upper_mode
            or "PASSING" in upper_mode
            or "CRITICAL" in upper_mode
            or "SLOW" in upper_mode
            or "STOP" in upper_mode
        )

        if is_intervention and not previous_is_intervention:
            event_count += 1

        previous_is_intervention = is_intervention

    return event_count


def calculate_metrics(
    method: MethodSpec,
    trial_index: int,
    csv_path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray | list[str]]]:
    rows = read_csv_rows(csv_path)

    if not rows:
        raise ValueError(
            f"Trajectory CSV is empty: {csv_path}"
        )

    time_values = np.array(
        [
            parse_float(
                first_existing(
                    row,
                    ("time_s", "time"),
                )
            )
            for row in rows
        ],
        dtype=float,
    )

    x_values = np.array(
        [
            parse_float(
                first_existing(row, ("x_m", "x"))
            )
            for row in rows
        ],
        dtype=float,
    )

    y_values = np.array(
        [
            parse_float(
                first_existing(row, ("y_m", "y"))
            )
            for row in rows
        ],
        dtype=float,
    )

    physical_clearance = np.array(
        [
            parse_float(
                first_existing(
                    row,
                    (
                        "physical_clearance_m",
                        "physical_clearance",
                    ),
                )
            )
            for row in rows
        ],
        dtype=float,
    )

    effective_clearance = np.array(
        [
            parse_float(
                first_existing(
                    row,
                    (
                        "effective_clearance_m",
                        "effective_clearance",
                    ),
                )
            )
            for row in rows
        ],
        dtype=float,
    )

    risk_values = np.array(
        [
            parse_float(
                first_existing(
                    row,
                    ("risk_score", "risk"),
                )
            )
            for row in rows
        ],
        dtype=float,
    )

    commanded_speed = np.array(
        [
            parse_float(
                first_existing(
                    row,
                    (
                        "commanded_speed_mps",
                        "speed",
                    ),
                )
            )
            for row in rows
        ],
        dtype=float,
    )

    detected_distance = np.array(
        [
            parse_float(
                first_existing(
                    row,
                    (
                        "nearest_detected_distance_m",
                        "estimated_distance_m",
                    ),
                )
            )
            for row in rows
        ],
        dtype=float,
    )

    detected_confidence = np.array(
        [
            parse_float(
                first_existing(
                    row,
                    (
                        "nearest_detected_confidence",
                        "confidence",
                    ),
                )
            )
            for row in rows
        ],
        dtype=float,
    )

    modes = [
        first_existing(
            row,
            ("control_mode", "mode"),
            default="UNKNOWN",
        )
        for row in rows
    ]

    final_position = np.array(
        [x_values[-1], y_values[-1]],
        dtype=float,
    )

    final_goal_error = float(
        np.linalg.norm(final_position - GOAL)
    )

    finite_physical = physical_clearance[
        np.isfinite(physical_clearance)
    ]
    finite_effective = effective_clearance[
        np.isfinite(effective_clearance)
    ]
    finite_risk = risk_values[
        np.isfinite(risk_values)
    ]
    finite_speed = commanded_speed[
        np.isfinite(commanded_speed)
    ]
    finite_confidence = detected_confidence[
        np.isfinite(detected_confidence)
    ]

    minimum_physical = (
        float(np.min(finite_physical))
        if finite_physical.size
        else float("nan")
    )
    minimum_effective = (
        float(np.min(finite_effective))
        if finite_effective.size
        else float("nan")
    )

    collision = (
        bool(minimum_physical <= 0.0)
        if math.isfinite(minimum_physical)
        else False
    )

    safety_violation = (
        bool(minimum_effective < 0.0)
        if math.isfinite(minimum_effective)
        else False
    )

    intervention_mask = np.array(
        [
            (
                "AVOID" in mode.upper()
                or "PASSING" in mode.upper()
                or "CRITICAL" in mode.upper()
                or "SLOW" in mode.upper()
                or "STOP" in mode.upper()
            )
            for mode in modes
        ],
        dtype=bool,
    )

    mission_time = (
        float(time_values[-1] - time_values[0])
        if len(time_values) >= 2
        else 0.0
    )

    if len(time_values) >= 2:
        sample_period = float(
            np.nanmedian(np.diff(time_values))
        )
    else:
        sample_period = 0.0

    detection_mask = np.isfinite(detected_distance)

    metric_row: dict[str, object] = {
        "method_key": method.key,
        "method": method.label,
        "trial": trial_index,
        "success": int(
            final_goal_error <= GOAL_TOLERANCE
            and not collision
        ),
        "collision": int(collision),
        "safety_margin_violation": int(
            safety_violation
        ),
        "final_goal_error_m": final_goal_error,
        "mission_time_s": mission_time,
        "path_length_m": calculate_path_length(
            x_values,
            y_values,
        ),
        "minimum_physical_clearance_m": (
            minimum_physical
        ),
        "minimum_effective_clearance_m": (
            minimum_effective
        ),
        "mean_commanded_speed_mps": (
            float(np.mean(finite_speed))
            if finite_speed.size
            else float("nan")
        ),
        "intervention_events": (
            count_intervention_events(modes)
        ),
        "intervention_time_s": float(
            np.sum(intervention_mask)
            * sample_period
        ),
        "intervention_fraction": float(
            np.mean(intervention_mask)
        ),
        "maximum_risk": (
            float(np.max(finite_risk))
            if finite_risk.size
            else float("nan")
        ),
        "mean_risk": (
            float(np.mean(finite_risk))
            if finite_risk.size
            else float("nan")
        ),
        "detection_rate": float(
            np.mean(detection_mask)
        ),
        "mean_detection_confidence": (
            float(np.mean(finite_confidence))
            if finite_confidence.size
            else float("nan")
        ),
        "trajectory_csv": str(csv_path),
    }

    series = {
        "time": time_values,
        "x": x_values,
        "y": y_values,
        "physical_clearance": physical_clearance,
        "effective_clearance": effective_clearance,
        "risk": risk_values,
        "modes": modes,
    }

    return metric_row, series


def write_dict_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(
    trial_metrics: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []

    numeric_metrics = (
        "success",
        "collision",
        "safety_margin_violation",
        "final_goal_error_m",
        "mission_time_s",
        "path_length_m",
        "minimum_physical_clearance_m",
        "minimum_effective_clearance_m",
        "mean_commanded_speed_mps",
        "intervention_events",
        "intervention_time_s",
        "intervention_fraction",
        "maximum_risk",
        "mean_risk",
        "detection_rate",
        "mean_detection_confidence",
    )

    for method in METHODS:
        rows = [
            row
            for row in trial_metrics
            if row["method_key"] == method.key
        ]

        if not rows:
            continue

        summary: dict[str, object] = {
            "method_key": method.key,
            "method": method.label,
            "trials": len(rows),
        }

        for metric in numeric_metrics:
            values = np.array(
                [
                    parse_float(row.get(metric))
                    for row in rows
                ],
                dtype=float,
            )

            finite = values[np.isfinite(values)]

            if finite.size:
                summary[f"{metric}_mean"] = float(
                    np.mean(finite)
                )
                summary[f"{metric}_std"] = float(
                    np.std(finite, ddof=0)
                )
            else:
                summary[f"{metric}_mean"] = float(
                    "nan"
                )
                summary[f"{metric}_std"] = float(
                    "nan"
                )

        summary_rows.append(summary)

    return summary_rows


def save_figure(
    figure: plt.Figure,
    output_root: Path,
    base_name: str,
) -> None:
    png_path = output_root / f"{base_name}.png"
    pdf_path = output_root / f"{base_name}.pdf"

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

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def plot_trajectory_comparison(
    series_by_method: dict[str, dict],
    output_root: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5.2))

    # Plot all method trajectories
    for method in METHODS:
        series = series_by_method.get(method.key)

        if series is None:
            continue

        x_values = np.asarray(series["x"], dtype=float)
        y_values = np.asarray(series["y"], dtype=float)

        finite = np.isfinite(x_values) & np.isfinite(y_values)

        axis.plot(
            x_values[finite],
            y_values[finite],
            linewidth=2.0,
            label=method.label,
        )

    # Use exactly the same map as the closed-loop camera plot
    obstacle_data = (
        ("O1", -0.10,  0.18, 0.20, "RIGHT"),
        ("O2",  1.15, -0.18, 0.20, "LEFT"),
        ("O3",  2.40,  0.18, 0.20, "RIGHT"),
        ("O4",  3.65, -0.18, 0.20, "LEFT"),
    )

    safety_margin = 0.18

    for name, x, y, radius, side in obstacle_data:

        # Physical obstacle
        axis.add_patch(
            plt.Circle(
                (x, y),
                radius,
                fill=True,
                alpha=0.30,
                zorder=1,
            )
        )

        # Required safety boundary
        axis.add_patch(
            plt.Circle(
                (x, y),
                radius + safety_margin,
                fill=False,
                linestyle="--",
                linewidth=1.8,
                zorder=1,
            )
        )

        axis.text(
            x,
            y,
            f"{name}\n{side}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )

    # Start and goal
    axis.scatter(
        [START[0]],
        [START[1]],
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

    axis.set_title("Safety-Oracle Trajectory Comparison")
    axis.set_xlabel("x position [m]")
    axis.set_ylabel("y position [m]")

    # Important: use the same limits as the nice individual plot
    axis.set_xlim(-1.45, 4.45)
    axis.set_ylim(-0.95, 0.95)

    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend(loc="upper right", fontsize=8)

    figure.tight_layout()

    save_figure(
        figure,
        output_root,
        "trajectory_comparison",
    )


def plot_clearance_over_time(
    series_by_method: dict[str, dict],
    output_root: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5.2))

    for method in METHODS:
        series = series_by_method.get(method.key)
        if series is None:
            continue

        clearance = np.asarray(
            series["physical_clearance"],
            dtype=float,
        )
        time_values = np.asarray(
            series["time"],
            dtype=float,
        )

        finite = (
            np.isfinite(clearance)
            & np.isfinite(time_values)
        )

        axis.plot(
            time_values[finite],
            clearance[finite],
            linewidth=1.8,
            label=method.label,
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Collision boundary",
    )
    axis.axhline(
        0.18,
        linestyle=":",
        linewidth=1.2,
        label="Safety-margin boundary",
    )

    axis.set_title(
        "Physical clearance over mission time"
    )
    axis.set_xlabel("time [s]")
    axis.set_ylabel(
        "nearest physical clearance [m]"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    save_figure(
        figure,
        output_root,
        "clearance_over_time",
    )


def plot_risk_vs_clearance(
    closed_loop_series: dict,
    output_root: Path,
) -> None:
    clearance = np.asarray(
        closed_loop_series["physical_clearance"],
        dtype=float,
    )
    risk = np.asarray(
        closed_loop_series["risk"],
        dtype=float,
    )

    finite = (
        np.isfinite(clearance)
        & np.isfinite(risk)
    )

    if not np.any(finite):
        print(
            "No finite risk data found; "
            "risk_vs_clearance figure skipped."
        )
        return

    clearance = clearance[finite]
    risk = risk[finite]

    order = np.argsort(clearance)

    figure, axis = plt.subplots(figsize=(8.2, 5.2))

    axis.scatter(
        clearance,
        risk,
        s=12,
        alpha=0.35,
        label="Camera-oracle samples",
    )

    # Bin the samples to show the average risk-clearance relationship.
    bin_edges = np.linspace(
        float(np.min(clearance)),
        float(np.max(clearance)),
        30,
    )

    bin_indices = np.digitize(
        clearance,
        bin_edges,
    )

    bin_centres: list[float] = []
    bin_means: list[float] = []

    for index in range(1, len(bin_edges)):
        mask = bin_indices == index

        if np.any(mask):
            bin_centres.append(
                float(np.mean(clearance[mask]))
            )
            bin_means.append(
                float(np.mean(risk[mask]))
            )

    if bin_centres:
        ordered_bins = np.argsort(
            np.asarray(bin_centres)
        )
        centres = np.asarray(
            bin_centres
        )[ordered_bins]
        means = np.asarray(
            bin_means
        )[ordered_bins]

        axis.plot(
            centres,
            means,
            linewidth=2.2,
            label="Binned mean risk",
        )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Collision boundary",
    )
    axis.axvline(
        0.18,
        linestyle=":",
        linewidth=1.2,
        label="Safety-margin boundary",
    )

    axis.set_title(
        "Camera-oracle risk versus obstacle clearance"
    )
    axis.set_xlabel(
        "nearest physical clearance [m]"
    )
    axis.set_ylabel("oracle risk score")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    save_figure(
        figure,
        output_root,
        "risk_vs_clearance",
    )


def plot_intervention_fraction(
    summary_rows: list[dict[str, object]],
    output_root: Path,
) -> None:
    methods = [
        str(row["method"])
        for row in summary_rows
    ]
    values = [
        parse_float(
            row.get(
                "intervention_fraction_mean"
            )
        )
        for row in summary_rows
    ]

    figure, axis = plt.subplots(figsize=(9.0, 5.0))

    positions = np.arange(len(methods))
    axis.bar(positions, values)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        methods,
        rotation=20,
        ha="right",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(
        "fraction of mission in intervention"
    )
    axis.set_title(
        "Safety-oracle intervention fraction"
    )
    axis.grid(True, axis="y")
    figure.tight_layout()

    save_figure(
        figure,
        output_root,
        "intervention_fraction",
    )


def print_summary(
    summary_rows: list[dict[str, object]],
) -> None:
    print("\nMethod comparison")

    header = (
        f"{'Method':40s} "
        f"{'Success':>8s} "
        f"{'Min clr':>10s} "
        f"{'Path':>9s} "
        f"{'Time':>9s} "
        f"{'Interv.':>9s}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{str(row['method']):40s} "
            f"{parse_float(row.get('success_mean')):8.2f} "
            f"{parse_float(row.get('minimum_physical_clearance_m_mean')):10.3f} "
            f"{parse_float(row.get('path_length_m_mean')):9.3f} "
            f"{parse_float(row.get('mission_time_s_mean')):9.2f} "
            f"{parse_float(row.get('intervention_fraction_mean')):9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all UAV oracle variants and generate "
            "paper-ready comparison metrics and figures."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help=(
            "Number of trials for each method. "
            "Use 20 or more for final paper statistics."
        ),
    )

    parser.add_argument(
        "--duration_sec",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="results/oracle_comparison",
    )

    parser.add_argument(
        "--python_executable",
        type=str,
        default=sys.executable,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print subprocess output in the terminal.",
    )

    parser.add_argument(
        "--skip_runs",
        action="store_true",
        help=(
            "Do not launch simulations. Analyse the newest "
            "trajectory CSV under each method folder."
        ),
    )

    arguments = parser.parse_args()

    if arguments.trials < 1:
        raise ValueError(
            "--trials must be at least 1"
        )

    comparison_root = (
        Path(arguments.output_root)
        / timestamp_string()
    )
    comparison_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    trial_metrics: list[dict[str, object]] = []
    first_series_by_method: dict[str, dict] = {}

    for method in METHODS:
        for trial_index in range(
            1,
            arguments.trials + 1,
        ):
            if arguments.skip_runs:
                method_root = Path(
                    arguments.output_root
                ) / method.key
                csv_path = find_trajectory_csv(
                    method_root
                )
            else:
                csv_path = run_method(
                    method=method,
                    trial_index=trial_index,
                    duration_sec=(
                        arguments.duration_sec
                    ),
                    comparison_root=(
                        comparison_root
                    ),
                    python_executable=(
                        arguments.python_executable
                    ),
                    verbose=arguments.verbose,
                )

            metric_row, series = calculate_metrics(
                method=method,
                trial_index=trial_index,
                csv_path=csv_path,
            )

            trial_metrics.append(metric_row)

            if method.key not in first_series_by_method:
                first_series_by_method[
                    method.key
                ] = series

    summary_rows = aggregate_metrics(
        trial_metrics
    )

    trial_csv = (
        comparison_root
        / "trial_metrics.csv"
    )
    summary_csv = (
        comparison_root
        / "method_summary.csv"
    )

    write_dict_csv(
        trial_csv,
        trial_metrics,
    )
    write_dict_csv(
        summary_csv,
        summary_rows,
    )

    print(f"\nSaved metrics: {trial_csv}")
    print(f"Saved summary: {summary_csv}")

    plot_trajectory_comparison(
        first_series_by_method,
        comparison_root,
    )

    plot_clearance_over_time(
        first_series_by_method,
        comparison_root,
    )

    closed_loop_series = (
        first_series_by_method.get(
            "closed_loop_camera"
        )
    )

    if closed_loop_series is not None:
        plot_risk_vs_clearance(
            closed_loop_series,
            comparison_root,
        )

    plot_intervention_fraction(
        summary_rows,
        comparison_root,
    )

    print_summary(summary_rows)

    print(
        f"\nAll comparison outputs are in: "
        f"{comparison_root}"
    )


if __name__ == "__main__":
    main()
