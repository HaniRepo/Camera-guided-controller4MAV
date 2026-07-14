from __future__ import annotations

import argparse
import time

import cv2

from perception import ColorObstaclePerception, PerceptionConfig


DEFAULT_STREAM_URL = "http://192.168.214.170:81/stream"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run color-obstacle perception using the real Crazyflie camera."
    )
    parser.add_argument(
        "--stream-url",
        default=DEFAULT_STREAM_URL,
    )
    parser.add_argument(
        "--horizontal-fov-deg",
        type=float,
        default=70.0,
    )
    parser.add_argument(
        "--obstacle-diameter-m",
        type=float,
        default=0.40,
    )
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.stream_url)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera stream: {args.stream_url}"
        )

    perception = ColorObstaclePerception(
        PerceptionConfig(
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            horizontal_fov_deg=args.horizontal_fov_deg,
            obstacle_diameter_m=args.obstacle_diameter_m,
        )
    )

    print("Real-camera perception started.")
    print("Press Q to stop.")

    while True:
        success, frame_bgr = capture.read()

        if not success or frame_bgr is None:
            print("Frame not received.")
            time.sleep(0.05)
            continue

        frame_bgr = cv2.resize(
            frame_bgr,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        observations = perception.process(
            rgb_frame=frame_rgb,
            timestamp_s=time.monotonic(),
        )

        annotated = perception.draw_detections(
            rgb_frame=frame_rgb,
            observations=observations,
        )

        if observations:
            nearest = observations[0]
            print(
                f"{nearest.obstacle_name}: "
                f"distance={nearest.estimated_distance_m:.2f} m, "
                f"bearing={nearest.bearing_rad:.3f} rad, "
                f"confidence={nearest.confidence:.2f}, "
                f"width={nearest.width_px:.0f} px"
            )

        cv2.imshow("Crazyflie Real Perception", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
