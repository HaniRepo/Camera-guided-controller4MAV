import time
import logging
import warnings

import cflib.crtp

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


URI = "radio://0/80/2M"


logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


def main():

    cflib.crtp.init_drivers(
        enable_debug_driver=False
    )

    print("Connecting...")

    with SyncCrazyflie(URI) as scf:

        print("Connected")

        try:

            # Important: arm motors
            print("Arming...")

            scf.cf.platform.send_arming_request(
                True
            )

            time.sleep(1)


            print("Takeoff")

            with MotionCommander(
                scf,
                default_height=0.5
            ) as mc:


                print("Hover")

                time.sleep(3)


                # Equivalent to nominal command
                # vx = forward
                # vy = left
                # vz = vertical

                print("Forward")

                mc.start_linear_motion(
                    0.10,   # forward m/s
                    0.0,    # left/right
                    0.0,    # vertical
                    0.0
                )

                time.sleep(3)



                print("Avoid right")

                mc.start_linear_motion(
                    0.08,    # forward
                    -0.10,   # right
                    0.0,
                    0.0
                )

                time.sleep(3)



                print("Stop")

                mc.stop()

                time.sleep(3)


                print("Landing")

                mc.land()


        finally:

            print("Stopping")

            try:
                scf.cf.commander.send_stop_setpoint()
                time.sleep(0.2)

            except Exception:
                pass


            print("Disarming")

            try:

                scf.cf.platform.send_arming_request(
                    False
                )

                time.sleep(1)

            except Exception:
                pass


    print("Finished")


if __name__ == "__main__":
    main()