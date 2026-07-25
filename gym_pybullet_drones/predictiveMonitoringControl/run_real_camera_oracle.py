from __future__ import annotations

import cv2
import numpy as np
import math
import time

from .real_camera import RealCamera
from .safety_oracle import (
    CameraSafetyOracle,
    SafetyOracleConfig,
)


# ==========================
# Camera calibration
# ==========================

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

FX = 160.0
KNOWN_WIDTH = 0.07


# ==========================
# Simple real perception
# (same output format as estimator)
# ==========================

class EstimatedObstacle:

    def __init__(
        self,
        name,
        distance,
        bearing,
        confidence=1.0,
    ):

        self.obstacle_name = name
        self.distance_m = distance
        self.forward_m = distance
        self.bearing_rad = math.radians(bearing)
        self.confidence = confidence
        self.frame_age_s = 0.0



def detect_red(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    mask1 = cv2.inRange(
        hsv,
        np.array([0,80,50]),
        np.array([10,255,255])
    )


    mask2 = cv2.inRange(
        hsv,
        np.array([170,80,50]),
        np.array([180,255,255])
    )


    mask = mask1 | mask2


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



def estimate_distance(width):

    return (
        FX *
        KNOWN_WIDTH
        /
        width
    )



def estimate_bearing(cx):

    center = IMAGE_WIDTH/2

    angle = math.atan2(
        cx-center,
        FX
    )

    return math.degrees(angle)



# ==========================
# MAIN
# ==========================


def main():


    camera = RealCamera(
        "http://192.168.0.6:81/stream"
    )


    oracle = CameraSafetyOracle(
        SafetyOracleConfig(
            warning_distance_m=0.8,
            critical_distance_m=0.35
        )
    )


    print(
        "Real camera oracle test started"
    )


    nominal_direction = np.array(
        [1.0,0.0]
    )


    try:

        while True:


            data = camera.capture()

            rgb = data["rgb"]


            frame = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )


            obstacle_states=[]


            detection = detect_red(
                frame
            )


            if detection:


                x,y,w,h=detection

                cx=x+w/2


                distance = estimate_distance(
                    w
                )


                bearing = estimate_bearing(
                    cx
                )


                obstacle_states.append(
                    EstimatedObstacle(
                        "RED_OBSTACLE",
                        distance,
                        bearing
                    )
                )


                cv2.rectangle(
                    frame,
                    (x,y),
                    (x+w,y+h),
                    (0,255,0),
                    2
                )


            decision = oracle.evaluate(
                nominal_direction,
                obstacle_states
            )


            text1 = (
                f"Mode: {decision.mode}"
            )

            text2 = (
                f"Dist: "
                f"{decision.estimated_distance_m:.2f} m"
            )

            text3 = (
                f"Bearing: "
                f"{math.degrees(decision.estimated_bearing_rad):.1f} deg"
            )

            text4 = (
                f"Risk: "
                f"{decision.risk_score:.2f}"
            )


            cv2.putText(
                frame,
                text1,
                (5,25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
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

            cv2.putText(
                frame,
                text3,
                (5,75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2
            )

            cv2.putText(
                frame,
                text4,
                (5,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2
            )


            cv2.imshow(
                "Real Camera Safety Oracle",
                frame
            )


            print(
                text1,
                "|",
                text2,
                "|",
                text3,
                "|",
                text4
            )


            if cv2.waitKey(1)==ord("q"):
                break


    finally:

        camera.close()
        cv2.destroyAllWindows()



if __name__=="__main__":

    main()