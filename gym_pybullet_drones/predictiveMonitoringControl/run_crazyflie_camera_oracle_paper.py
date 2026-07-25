"""Paper-quality Crazyflie camera-oracle experiment.

Fixed obstacle placement is allowed for repeatability, but obstacle coordinates
are not used by this controller. Press q or Ctrl+C to stop safely.
"""
from __future__ import annotations

import csv
import logging
import math
import threading
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import cflib.crtp
import numpy as np
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = "radio://0/80/2M"
STREAM_URL = "http://192.168.0.6:81/stream"
FLIGHT_HEIGHT_M = 0.22
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
FX_PX = 160.0
OBJECT_WIDTH_M = 0.07
WARNING_DISTANCE_M = 0.80
CRITICAL_DISTANCE_M = 0.35
NOMINAL_VX_MPS = 0.05
AVOID_VX_MPS = 0.02
AVOID_VY_MPS = 0.025
CRITICAL_VY_MPS = 0.035
RECOVERY_VX_MPS = 0.035
RECOVERY_VY_MPS = 0.012
RECOVERY_DURATION_S = 1.6
FILTER_SIZE = 5
MIN_CONTOUR_AREA_PX = 50.0
BEARING_DEADBAND_DEG = 3.0
MISSING_FRAMES_BEFORE_RECOVERY = 5
MAX_FLIGHT_TIME_S = 90.0

RUN_DIRECTORY = Path("results") / "crazyflie_hardware" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
CSV_PATH = RUN_DIRECTORY / "oracle_log.csv"
VIDEO_PATH = RUN_DIRECTORY / "annotated_camera.mp4"
FIRST_FRAME_PATH = RUN_DIRECTORY / "first_annotated_frame.png"

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")


class LatestFrameCamera:
    """Continuously reads the MJPEG stream and keeps only the newest frame."""

    def __init__(self, url: str, width: int, height: int) -> None:
        self.url = url
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_time = 0.0
        self.running = False

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Cannot open camera stream: {self.url}")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            if self.cap is None:
                time.sleep(0.05)
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, (self.width, self.height))
            with self.lock:
                self.frame = frame
                self.frame_time = time.monotonic()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self.lock:
            if self.frame is None:
                return None, float("inf")
            return self.frame.copy(), time.monotonic() - self.frame_time

    def close(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class OracleMemory:
    def __init__(self) -> None:
        self.active_side: str | None = None
        self.missing_frames = 0
        self.recovery_start_s: float | None = None
        self.encounter_id = 0
        self.in_encounter = False


def detect_red(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255]))
    high = cv2.inRange(hsv, np.array([170, 80, 50]), np.array([180, 255, 255]))
    mask = low | high
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA_PX:
        return None
    return cv2.boundingRect(contour)


def estimate_distance(width_px: int) -> float:
    return float("inf") if width_px <= 0 else FX_PX * OBJECT_WIDTH_M / width_px


def estimate_bearing(center_x_px: float) -> float:
    return math.degrees(math.atan2(center_x_px - IMAGE_WIDTH / 2.0, FX_PX))


def calculate_risk(distance_m: float) -> float:
    if not math.isfinite(distance_m):
        return 0.0
    return float(np.clip((WARNING_DISTANCE_M - distance_m) / (WARNING_DISTANCE_M - CRITICAL_DISTANCE_M), 0.0, 1.0))


def choose_side(bearing_deg: float) -> str:
    if bearing_deg < -BEARING_DEADBAND_DEG:
        return "RIGHT"
    if bearing_deg > BEARING_DEADBAND_DEG:
        return "LEFT"
    return "RIGHT"


def oracle(visible: bool, distance: float, bearing: float, memory: OracleMemory, now: float):
    risk = calculate_risk(distance)
    if visible and distance <= WARNING_DISTANCE_M:
        memory.missing_frames = 0
        memory.recovery_start_s = None
        if not memory.in_encounter:
            memory.encounter_id += 1
            memory.in_encounter = True
        if memory.active_side is None:
            memory.active_side = choose_side(bearing)
        sign = -1.0 if memory.active_side == "RIGHT" else 1.0
        if distance <= CRITICAL_DISTANCE_M:
            return 0.0, sign * CRITICAL_VY_MPS, f"CRITICAL_AVOID_{memory.active_side}", risk
        return AVOID_VX_MPS, sign * AVOID_VY_MPS, f"AVOID_{memory.active_side}", risk

    if memory.active_side is not None:
        memory.missing_frames += 1
        if memory.missing_frames < MISSING_FRAMES_BEFORE_RECOVERY:
            sign = -1.0 if memory.active_side == "RIGHT" else 1.0
            return AVOID_VX_MPS, sign * AVOID_VY_MPS, f"PASSING_{memory.active_side}", 0.0
        if memory.recovery_start_s is None:
            memory.recovery_start_s = now
            memory.in_encounter = False
        if now - memory.recovery_start_s < RECOVERY_DURATION_S:
            sign = 1.0 if memory.active_side == "RIGHT" else -1.0
            mode = "RECOVER_LEFT" if memory.active_side == "RIGHT" else "RECOVER_RIGHT"
            return RECOVERY_VX_MPS, sign * RECOVERY_VY_MPS, mode, 0.0
        memory.active_side = None
        memory.missing_frames = 0
        memory.recovery_start_s = None

    return NOMINAL_VX_MPS, 0.0, "NOMINAL", 0.0


def draw_overlay(frame, bbox, distance, bearing, risk, mode, vx, vy, frame_age, encounter):
    display = frame.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    display = cv2.copyMakeBorder(display, 105, 0, 0, 0, cv2.BORDER_CONSTANT, value=(22, 22, 22))
    distance_text = f"{distance:.2f} m" if math.isfinite(distance) else "none"
    lines = [
        f"Mode: {mode}   Encounter: {encounter}",
        f"Distance: {distance_text}   Bearing: {bearing:.1f} deg   Risk: {risk:.2f}",
        f"Command: vx={vx:.3f} m/s, vy={vy:.3f} m/s   Frame age: {frame_age:.3f} s",
    ]
    for i, text in enumerate(lines):
        cv2.putText(display, text, (8, 27 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return display


def shutdown(scf, mc, camera, csv_file, video):
    print("Stopping safely...")
    try:
        if mc is not None:
            mc.stop()
            time.sleep(0.2)
            mc.land(velocity=0.12)
    except Exception:
        pass
    try:
        scf.cf.commander.send_stop_setpoint()
    except Exception:
        pass
    try:
        scf.cf.platform.send_arming_request(False)
    except Exception:
        pass
    camera.close()
    try:
        csv_file.flush()
        csv_file.close()
    except Exception:
        pass
    if video is not None:
        video.release()
    cv2.destroyAllWindows()
    print("Finished safely.")


def main() -> None:
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {RUN_DIRECTORY}")
    csv_file = CSV_PATH.open("w", newline="", encoding="utf-8")
    fields = [
        "time_s", "frame_age_s", "obstacle_visible", "encounter_id",
        "distance_raw_m", "distance_filtered_m", "bearing_raw_deg",
        "bearing_filtered_deg", "risk", "mode", "vx_mps", "vy_mps",
        "command_integrated_x_m", "command_integrated_y_m",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    camera = LatestFrameCamera(STREAM_URL, IMAGE_WIDTH, IMAGE_HEIGHT)
    camera.start()
    cflib.crtp.init_drivers(enable_debug_driver=False)
    distances = deque(maxlen=FILTER_SIZE)
    bearings = deque(maxlen=FILTER_SIZE)
    memory = OracleMemory()
    cv2.namedWindow("Crazyflie Camera Oracle", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crazyflie Camera Oracle", 800, 600)
    video = None
    mc = None

    with SyncCrazyflie(URI) as scf:
        try:
            scf.cf.platform.send_arming_request(True)
            time.sleep(1.0)
            mc = MotionCommander(scf, default_height=FLIGHT_HEIGHT_M)
            mc.take_off()
            time.sleep(2.0)
            start = time.monotonic()
            previous = start
            integrated_x = 0.0
            integrated_y = 0.0
            saved_first = False
            last_frame_time = -1.0

            while True:
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= MAX_FLIGHT_TIME_S:
                    print("Maximum experiment time reached.")
                    break
                frame, age = camera.latest()
                if frame is None:
                    time.sleep(0.01)
                    continue
                frame_time = now - age
                if frame_time <= last_frame_time:
                    time.sleep(0.005)
                    continue
                last_frame_time = frame_time

                bbox = detect_red(frame)
                visible = bbox is not None
                d_raw = float("nan")
                b_raw = float("nan")
                d = float("inf")
                b = 0.0
                if bbox is not None:
                    x, y, w, h = bbox
                    d_raw = estimate_distance(w)
                    b_raw = estimate_bearing(x + w / 2.0)
                    distances.append(d_raw)
                    bearings.append(b_raw)
                    d = float(np.median(distances))
                    b = float(np.median(bearings))
                else:
                    distances.clear()
                    bearings.clear()

                vx, vy, mode, risk = oracle(visible, d, b, memory, now)
                mc.start_linear_motion(vx, vy, 0.0, 0.0)
                dt = max(0.0, now - previous)
                previous = now
                integrated_x += vx * dt
                integrated_y += vy * dt

                writer.writerow({
                    "time_s": f"{elapsed:.6f}",
                    "frame_age_s": f"{age:.6f}",
                    "obstacle_visible": int(visible),
                    "encounter_id": memory.encounter_id,
                    "distance_raw_m": f"{d_raw:.6f}" if math.isfinite(d_raw) else "",
                    "distance_filtered_m": f"{d:.6f}" if math.isfinite(d) else "",
                    "bearing_raw_deg": f"{b_raw:.6f}" if math.isfinite(b_raw) else "",
                    "bearing_filtered_deg": f"{b:.6f}",
                    "risk": f"{risk:.6f}",
                    "mode": mode,
                    "vx_mps": f"{vx:.6f}",
                    "vy_mps": f"{vy:.6f}",
                    "command_integrated_x_m": f"{integrated_x:.6f}",
                    "command_integrated_y_m": f"{integrated_y:.6f}",
                })
                csv_file.flush()

                display = draw_overlay(frame, bbox, d, b, risk, mode, vx, vy, age, memory.encounter_id)
                if video is None:
                    h, w = display.shape[:2]
                    video = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                if not saved_first:
                    cv2.imwrite(str(FIRST_FRAME_PATH), display)
                    saved_first = True
                video.write(display)
                cv2.imshow("Crazyflie Camera Oracle", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("Ctrl+C pressed.")
        finally:
            shutdown(scf, mc, camera, csv_file, video)

    print(f"CSV saved: {CSV_PATH}")
    print(f"Video saved: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
