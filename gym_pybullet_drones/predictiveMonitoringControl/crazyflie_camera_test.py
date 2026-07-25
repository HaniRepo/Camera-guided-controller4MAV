from __future__ import annotations

import cv2
import numpy as np
import math
import time


# ==========================
# Camera parameters
# ==========================

STREAM_URL = "http://192.168.0.6:81/stream"

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

# calibrated values from your experiment
FX = 160.0

# your red obstacle width
KNOWN_OBJECT_WIDTH_M = 0.07


# Safety oracle thresholds
WARNING_DISTANCE = 0.80
CRITICAL_DISTANCE = 0.35



# ==========================
# Camera
# ==========================

class RealCamera:

    def __init__(self):

        self.cap = cv2.VideoCapture(
            STREAM_URL
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot connect to camera"
            )

        print("Camera connected")


    def capture(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        frame = cv2.resize(
            frame,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        return frame


    def close(self):
        self.cap.release()



# ==========================
# Perception
# ==========================

def detect_red_obstacle(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    lower1 = np.array(
        [0,80,50]
    )

    upper1 = np.array(
        [10,255,255]
    )


    lower2 = np.array(
        [170,80,50]
    )

    upper2 = np.array(
        [180,255,255]
    )


    mask = (
        cv2.inRange(
            hsv,
            lower1,
            upper1
        )
        |
        cv2.inRange(
            hsv,
            lower2,
            upper2
        )
    )


    kernel = np.ones(
        (5,5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours)==0:
        return None


    c=max(
        contours,
        key=cv2.contourArea
    )


    area=cv2.contourArea(c)


    if area < 50:
        return None


    x,y,w,h=cv2.boundingRect(c)


    return x,y,w,h



# ==========================
# Estimation
# ==========================

def estimate_distance(pixel_width):

    if pixel_width <= 0:
        return None

    distance = (
        FX *
        KNOWN_OBJECT_WIDTH_M
        /
        pixel_width
    )

    return distance



def estimate_bearing(cx):

    image_center = IMAGE_WIDTH/2

    angle = math.atan2(
        cx-image_center,
        FX
    )

    return math.degrees(angle)



# ==========================
# Safety oracle (temporary)
# ==========================

def compute_risk(distance):

    if distance > WARNING_DISTANCE:
        return 0.0

    risk = (
        WARNING_DISTANCE-distance
    ) / (
        WARNING_DISTANCE-CRITICAL_DISTANCE
    )

    return float(
        np.clip(
            risk,
            0,
            1
        )
    )


def oracle_mode(distance):

    if distance > WARNING_DISTANCE:
        return "NOMINAL"

    elif distance > CRITICAL_DISTANCE:
        return "AVOID"

    else:
        return "CRITICAL_AVOID"



# ==========================
# Main
# ==========================

camera = RealCamera()


print(
    "Starting camera safety test"
)


try:

    while True:

        frame = camera.capture()


        if frame is None:
            continue


        detection = detect_red_obstacle(
            frame
        )


        if detection:

            x,y,w,h = detection


            cx = x+w/2


            distance = estimate_distance(
                w
            )


            bearing = estimate_bearing(
                cx
            )


            risk = compute_risk(
                distance
            )


            mode = oracle_mode(
                distance
            )


            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )


            text1 = (
                f"D={distance:.2f}m "
                f"B={bearing:.1f}deg"
            )

            text2 = (
                f"Risk={risk:.2f} "
                f"Mode={mode}"
            )


            cv2.putText(
                frame,
                text1,
                (5,25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )


            cv2.putText(
                frame,
                text2,
                (5,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2
            )


            print(
                text1,
                "|",
                text2
            )


        else:

            cv2.putText(
                frame,
                "No obstacle",
                (5,25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )


        cv2.imshow(
            "Crazyflie Real Camera Test",
            frame
        )


        key=cv2.waitKey(1)

        if key==ord("q"):
            break


finally:

    camera.close()
    cv2.destroyAllWindows()