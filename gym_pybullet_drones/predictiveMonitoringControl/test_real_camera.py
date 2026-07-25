import cv2

from real_camera import RealCamera


camera = RealCamera()


while True:

    output = camera.capture()

    rgb = output["rgb"]

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow(
        "OV3660 Real Camera",
        bgr
    )

    if cv2.waitKey(1) == ord("q"):
        break


camera.close()
cv2.destroyAllWindows()