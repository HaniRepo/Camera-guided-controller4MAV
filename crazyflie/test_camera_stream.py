from __future__ import annotations

import argparse
import time

import cv2


DEFAULT_STREAM_URL = "http://192.168.214.170:81/stream"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display the XIAO ESP32S3 camera stream mounted on the Crazyflie."
    )
    parser.add_argument(
        "--stream-url",
        default=DEFAULT_STREAM_URL,
        help="MJPEG stream URL from the XIAO ESP32S3 camera.",
    )
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.stream_url)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera stream: {args.stream_url}"
        )

    print(f"Camera stream opened: {args.stream_url}")
    print("Press Q to close.")

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    while True:
        success, frame = capture.read()

        if not success or frame is None:
            print("Frame not received.")
            time.sleep(0.05)
            continue

        now = time.perf_counter()
        elapsed = now - previous_time
        previous_time = now

        instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        smoothed_fps = (
            instantaneous_fps
            if smoothed_fps == 0.0
            else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
        )

        cv2.putText(
            frame,
            f"FPS: {smoothed_fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Crazyflie Real Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
