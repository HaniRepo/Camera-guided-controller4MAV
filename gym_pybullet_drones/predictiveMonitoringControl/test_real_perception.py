import cv2
import numpy as np
import math

from real_camera import RealCamera


# Camera parameters
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

# Approximate focal length (will calibrate later)
FX =  160


# Real obstacle size assumption
# Example: red cylinder diameter 7 cm
KNOWN_OBJECT_WIDTH_M =  0.07


def detect_red_object(frame_bgr):

    hsv = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2HSV
    )

    # red range 1
    lower_red1 = np.array(
        [0, 80, 50]
    )

    upper_red1 = np.array(
        [10, 255, 255]
    )

    # red range 2
    lower_red2 = np.array(
        [170, 80, 50]
    )

    upper_red2 = np.array(
        [180, 255, 255]
    )


    mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    mask = mask1 | mask2


    # remove noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((5,5),np.uint8)
    )


    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours)==0:
        return None


    contour=max(
        contours,
        key=cv2.contourArea
    )


    area=cv2.contourArea(contour)

    if area < 100:
        return None


    x,y,w,h=cv2.boundingRect(contour)


    return x,y,w,h



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



def compute_bearing(cx):

    image_center = IMAGE_WIDTH/2

    angle = math.atan2(
        cx-image_center,
        FX
    )

    return math.degrees(angle)



camera = RealCamera(
    "http://192.168.0.6:81/stream"
)


print("Starting real perception test")


while True:


    data=camera.capture()

    rgb=data["rgb"]


    frame=cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


    detection=detect_red_object(
        frame
    )


    if detection:

        x,y,w,h=detection


        cx=x+w/2


        distance=estimate_distance(
            w
        )


        bearing=compute_bearing(
            cx
        )


        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )


        text=f"D={distance:.2f}m  B={bearing:.1f}deg"


        cv2.putText(
            frame,
            text,
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


        print(text)


    else:

        cv2.putText(
            frame,
            "No obstacle",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )


    cv2.imshow(
        "Real camera perception",
        frame
    )


    if cv2.waitKey(1)==ord("q"):
        break



camera.close()
cv2.destroyAllWindows()