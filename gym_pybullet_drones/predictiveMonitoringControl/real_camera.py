from __future__ import annotations

import cv2
import numpy as np


class RealCamera:
    """
    Real OV3660 camera interface.
    Provides the same type of RGB frame as VirtualCamera.
    """

    def __init__(
        self,
        stream_url: str = "http://192.168.0.6:81/stream",
        width: int = 320,
        height: int = 240,
    ):

        self.stream_url = stream_url
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(
            self.stream_url
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot connect to OV3660 camera"
            )

        print(
            "Real camera connected:",
            self.stream_url
        )


    def capture(self):

        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError(
                "Failed to read camera frame"
            )

        frame = cv2.resize(
            frame,
            (self.width, self.height)
        )

        # OpenCV gives BGR
        # Our pipeline uses RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        return {
            "rgb": rgb
        }


    def close(self):

        self.cap.release()