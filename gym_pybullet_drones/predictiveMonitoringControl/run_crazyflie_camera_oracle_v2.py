import time
import logging
import warnings
import math
import os
import csv
from datetime import datetime

import cv2
import numpy as np
import cflib.crtp

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


URI = "radio://0/80/2M"

STREAM_URL = "http://192.168.0.6:81/stream"


# ==========================
# Camera calibration
# ==========================

FX = 160.0
OBJECT_WIDTH = 0.07


# ==========================
# Safety distances
# ==========================

WARNING_DISTANCE = 0.80
CRITICAL_DISTANCE = 0.35


# ==========================
# Logging
# ==========================

RESULT_DIR = os.path.join(
    "results",
    "crazyflie_hardware",
    datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


LOG_FILE = os.path.join(
    RESULT_DIR,
    "oracle_log.csv"
)



# ==========================
# Controller memory
# ==========================

last_avoid_side = None
recover_counter = 0


# filtering
bearing_history = []
distance_history = []



logging.basicConfig(
    level=logging.ERROR
)

warnings.filterwarnings(
    "ignore"
)



# ==========================
# Perception
# ==========================

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


    if not contours:
        return None


    c=max(
        contours,
        key=cv2.contourArea
    )


    if cv2.contourArea(c)<50:
        return None


    return cv2.boundingRect(c)



def estimate(width):

    return (
        FX *
        OBJECT_WIDTH /
        width
    )



def bearing(cx):

    return math.degrees(
        math.atan2(
            cx-160,
            FX
        )
    )



def smooth_distance(value):

    distance_history.append(
        value
    )

    if len(distance_history)>5:
        distance_history.pop(0)


    return (
        sum(distance_history)
        /
        len(distance_history)
    )



def smooth_bearing(value):

    bearing_history.append(
        value
    )

    if len(bearing_history)>5:
        bearing_history.pop(0)


    return (
        sum(bearing_history)
        /
        len(bearing_history)
    )



# ==========================
# Safety Oracle
# ==========================

def oracle(distance, angle):

    global last_avoid_side
    global recover_counter


    risk = np.clip(
        (
            WARNING_DISTANCE-distance
        )
        /
        (
            WARNING_DISTANCE-CRITICAL_DISTANCE
        ),
        0,
        1
    )



    # no obstacle / recovery

    if distance > WARNING_DISTANCE:


        if last_avoid_side is not None:

            recover_counter += 1


            if recover_counter < 30:


                if last_avoid_side=="RIGHT":

                    return (
                        0.05,
                        0.015,
                        "RECOVER_LEFT",
                        risk
                    )


                else:

                    return (
                        0.05,
                        -0.015,
                        "RECOVER_RIGHT",
                        risk
                    )


            else:

                last_avoid_side=None
                recover_counter=0



        return (
            0.05,
            0.0,
            "NOMINAL",
            risk
        )



    # critical

    if distance < CRITICAL_DISTANCE:


        if angle < 0:

            last_avoid_side="RIGHT"

            return (
                0.0,
                -0.035,
                "CRITICAL_AVOID_RIGHT",
                risk
            )


        else:

            last_avoid_side="LEFT"

            return (
                0.0,
                0.035,
                "CRITICAL_AVOID_LEFT",
                risk
            )



    # normal avoidance

    if angle < 0:

        last_avoid_side="RIGHT"

        return (
            0.02,
            -0.025,
            "AVOID_RIGHT",
            risk
        )


    else:

        last_avoid_side="LEFT"

        return (
            0.02,
            0.025,
            "AVOID_LEFT",
            risk
        )
    # ==========================
# Safe shutdown
# ==========================

def safe_shutdown(
    scf,
    mc,
    cap,
    log_file
):

    print("Stopping safely...")


    try:
        mc.stop()
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
        time.sleep(1)
    except Exception:
        pass


    try:
        cap.release()
    except Exception:
        pass


    try:
        log_file.close()
    except Exception:
        pass


    cv2.destroyAllWindows()

    print("Finished")



# ==========================
# Main
# ==========================

def main():


    print(
        "Saving results to:",
        RESULT_DIR
    )


    log_file = open(
        LOG_FILE,
        "w",
        newline=""
    )


    writer = csv.writer(
        log_file
    )


    writer.writerow(
        [
            "time",
            "distance_m",
            "bearing_deg",
            "risk",
            "mode",
            "vx",
            "vy"
        ]
    )


    cflib.crtp.init_drivers(
        enable_debug_driver=False
    )



    cap=cv2.VideoCapture(
        STREAM_URL
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Camera stream not available"
        )


    print("Camera connected")



    cv2.namedWindow(
        "Crazyflie Camera Oracle",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Crazyflie Camera Oracle",
        640,
        480
    )



    print(
        "Connecting Crazyflie..."
    )


    with SyncCrazyflie(URI) as scf:


        print(
            "Crazyflie connected"
        )


        scf.cf.platform.send_arming_request(
            True
        )


        time.sleep(1)



        with MotionCommander(
            scf,
            default_height=0.22
        ) as mc:


            print(
                "Takeoff"
            )


            time.sleep(3)



            start_time=time.time()



            try:

                while True:


                    ret,frame=cap.read()


                    if not ret:

                        continue



                    vx=0.05
                    vy=0.0

                    mode="NOMINAL"

                    distance=99.0
                    angle=0.0
                    risk=0.0



                    obj=detect_red(frame)



                    if obj:


                        x,y,w,h=obj



                        distance_raw=estimate(w)

                        distance=smooth_distance(
                            distance_raw
                        )


                        angle_raw=bearing(
                            x+w/2
                        )

                        angle=smooth_bearing(
                            angle_raw
                        )



                        (
                            vx,
                            vy,
                            mode,
                            risk
                        ) = oracle(
                            distance,
                            angle
                        )



                        cv2.rectangle(
                            frame,
                            (x,y),
                            (x+w,y+h),
                            (0,255,0),
                            2
                        )



                    else:


                        (
                            vx,
                            vy,
                            mode,
                            risk
                        ) = oracle(
                            99,
                            0
                        )



                    # Send command

                    mc.start_linear_motion(
                        vx,
                        vy,
                        0,
                        0
                    )



                    # Save experiment data

                    writer.writerow(
                        [
                            time.time()-start_time,
                            distance,
                            angle,
                            risk,
                            mode,
                            vx,
                            vy
                        ]
                    )


                    log_file.flush()



                    print(
                        f"D={distance:.2f} m | "
                        f"B={angle:.1f} deg | "
                        f"Risk={risk:.2f} | "
                        f"{mode} | "
                        f"vx={vx:.2f} "
                        f"vy={vy:.2f}"
                    )



                    # Display

                    cv2.putText(
                        frame,
                        f"MODE={mode}",
                        (5,25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"D={distance:.2f}m Risk={risk:.2f}",
                        (5,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,0),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"vx={vx:.2f} vy={vy:.2f}",
                        (5,75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2
                    )



                    cv2.imshow(
                        "Crazyflie Camera Oracle",
                        frame
                    )



                    key=cv2.waitKey(1)



                    if key==ord("q"):

                        print(
                            "Q pressed"
                        )

                        break



            except KeyboardInterrupt:

                print(
                    "Ctrl+C pressed"
                )



            finally:


                safe_shutdown(
                    scf,
                    mc,
                    cap,
                    log_file
                )



if __name__=="__main__":

    main()