# Predictive Safety Monitoring and Obstacle Avoidance for Crazyflie MAVs

This repository contains the implementation of a lightweight predictive safety monitoring framework for micro aerial vehicle (MAV) obstacle avoidance.

The framework combines obstacle perception, distance/bearing estimation, risk assessment, and a command-level safety oracle to modify the drone motion before collision occurs. The approach is designed for resource-constrained platforms such as the Bitcraze Crazyflie.

## Demo
Our Demo also won IEEE SYPA award on IECON 2026 Conference. You can have a loot to this demo at the IES official YouTube Channel at the following link:
[Demo](https://www.youtube.com/watch?v=-71jVl92HJ4)

## Overview

The proposed framework follows the pipeline:

```text
Virtual Camera → Perception → State Estimation → Safety Oracle → UAV Command
```
## The safety oracle operates in four modes:

- **Nominal:** Continue forward navigation when obstacles are far away.
- **Avoidance:** Reduce forward velocity and apply lateral motion.
- **Critical Avoidance:** Stop forward motion and apply stronger lateral correction.
- **Recovery:** Return toward the nominal navigation direction after obstacle clearance.

---

## Features

- PyBullet-based MAV simulation using `gym-pybullet-drones`
- Crazyflie 2.1 hardware validation
- Camera-based obstacle detection
- Monocular distance estimation
- Bearing estimation from image coordinates
- Risk-based safety monitoring
- Command-level obstacle avoidance
- Trajectory, clearance, risk, and command logging
- Automatic generation of experimental figures

---

## Hardware Platform

Experiments were conducted using:

- Bitcraze Crazyflie 2.1 Brushless
- Flow Deck for low-altitude stabilization
- Crazyradio communication link
- External camera stream for perception experiments

The hardware workspace consists of a compact indoor environment with two colored obstacles. The drone performs low-altitude obstacle avoidance followed by automatic landing.

---



## Run the comparison
```bashe
python -m gym_pybullet_drones.predictiveMonitoringControl.compare_oracle_experiments --trials 1 --duration_sec 200
```
Generate paper figures
```bashe
python -m gym_pybullet_drones.predictiveMonitoringControl.plot_paper_diagnostics \
    --comparison_dir results/oracle_comparison/<run_folder>
```
The generated results include trajectory, clearance, risk, speed, intervention metrics, and per-obstacle diagnostic figures.

## Run the Sensitivity Analysis Test
```bashe
python -m gym_pybullet_drones.predictiveMonitoringControl.run_camera_oracle_sensitivity
```
It runs seven experiments automatically:
```text
nominal
threshold -10%
threshold +10%
speed -10%
speed +10%
lateral weight -10%
lateral weight +10%
```

## Running Crazyflie Hardware

The closed-loop camera oracle successfully avoids four coloured obstacles and reaches the goal without using ground-truth obstacle coordinates for control.
```bashe
python -m gym_pybullet_drones.predictiveMonitoringControl.run_crazyflie_camera_oracle
```
The system records:

- obstacle distance
- obstacle bearing
- risk value
- operating mode
- velocity commands
- annotated camera frames

### Safety Parameters

The main parameters controlling the safety response are:
```text
WARNING_DISTANCE = 0.30   # meters

CRITICAL_DISTANCE = 0.12  # meters
```

### Experimental Outputs
Each experiment generates a timestamped folder:
```text
results/
└── run_xxxxxx/
    ├── log.csv
    ├── trajectory.png
    ├── clearance.png
    ├── risk.png
    ├── commands.png
    └── video.mp4
```
