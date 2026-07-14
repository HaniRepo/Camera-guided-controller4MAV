from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class ObstacleObservation:
    obstacle_name: str
    center_x_px: float
    center_y_px: float
    width_px: float
    height_px: float
    bearing_rad: float
    estimated_distance_m: float
    confidence: float
    timestamp_s: float
    bounding_box: tuple[int, int, int, int]


@dataclass
class PerceptionConfig:
    image_width: int = 320
    image_height: int = 240
    horizontal_fov_deg: float = 70.0

    obstacle_diameter_m: float = 0.40

    minimum_contour_area_px: float = 120.0
    minimum_width_px: float = 6.0

    morphology_kernel_size: int = 5

    red_min_1: tuple[int, int, int] = (0, 100, 80)
    red_max_1: tuple[int, int, int] = (10, 255, 255)
    red_min_2: tuple[int, int, int] = (170, 100, 80)
    red_max_2: tuple[int, int, int] = (179, 255, 255)

    blue_min: tuple[int, int, int] = (95, 90, 70)
    blue_max: tuple[int, int, int] = (135, 255, 255)

    yellow_min: tuple[int, int, int] = (20, 100, 100)
    yellow_max: tuple[int, int, int] = (38, 255, 255)

    green_min: tuple[int, int, int] = (40, 70, 60)
    green_max: tuple[int, int, int] = (90, 255, 255)


class ColorObstaclePerception:
    """
    Detect coloured cylindrical obstacles from RGB camera frames.

    The controller must later use only these observations.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
    ) -> None:
        self.config = config or PerceptionConfig()

        horizontal_fov_rad = np.deg2rad(
            self.config.horizontal_fov_deg
        )

        self.focal_length_px = (
            self.config.image_width
            / (
                2.0
                * np.tan(horizontal_fov_rad / 2.0)
            )
        )

        kernel_size = self.config.morphology_kernel_size

        self.kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

    def _build_colour_masks(
        self,
        hsv_image: np.ndarray,
    ) -> dict[str, np.ndarray]:
        red_mask_1 = cv2.inRange(
            hsv_image,
            np.array(self.config.red_min_1),
            np.array(self.config.red_max_1),
        )

        red_mask_2 = cv2.inRange(
            hsv_image,
            np.array(self.config.red_min_2),
            np.array(self.config.red_max_2),
        )

        red_mask = cv2.bitwise_or(
            red_mask_1,
            red_mask_2,
        )

        blue_mask = cv2.inRange(
            hsv_image,
            np.array(self.config.blue_min),
            np.array(self.config.blue_max),
        )

        yellow_mask = cv2.inRange(
            hsv_image,
            np.array(self.config.yellow_min),
            np.array(self.config.yellow_max),
        )

        green_mask = cv2.inRange(
            hsv_image,
            np.array(self.config.green_min),
            np.array(self.config.green_max),
        )

        return {
            "O1_RED": red_mask,
            "O2_BLUE": blue_mask,
            "O3_YELLOW": yellow_mask,
            "O4_GREEN": green_mask,
        }

    def _clean_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        cleaned_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
        )

        cleaned_mask = cv2.morphologyEx(
            cleaned_mask,
            cv2.MORPH_CLOSE,
            self.kernel,
        )

        return cleaned_mask

    def _estimate_bearing(
        self,
        center_x_px: float,
    ) -> float:
        principal_point_x = (
            self.config.image_width / 2.0
        )

        return float(
            np.arctan(
                (
                    center_x_px
                    - principal_point_x
                )
                / self.focal_length_px
            )
        )

    def _estimate_distance(
        self,
        width_px: float,
    ) -> float:
        if width_px <= 0.0:
            return float("inf")

        return float(
            (
                self.focal_length_px
                * self.config.obstacle_diameter_m
            )
            / width_px
        )

    def _estimate_confidence(
        self,
        contour_area: float,
        bounding_box_area: float,
        width_px: float,
    ) -> float:
        if bounding_box_area <= 0.0:
            return 0.0

        fill_ratio = contour_area / bounding_box_area

        width_score = np.clip(
            width_px / 80.0,
            0.0,
            1.0,
        )

        area_score = np.clip(
            contour_area / 2500.0,
            0.0,
            1.0,
        )

        confidence = (
            0.45 * fill_ratio
            + 0.30 * width_score
            + 0.25 * area_score
        )

        return float(
            np.clip(confidence, 0.0, 1.0)
        )

    def process(
        self,
        rgb_frame: np.ndarray,
        timestamp_s: float,
    ) -> list[ObstacleObservation]:
        if rgb_frame is None:
            return []

        if rgb_frame.ndim != 3:
            raise ValueError(
                "rgb_frame must have shape "
                "(height, width, channels)"
            )

        frame_height, frame_width = rgb_frame.shape[:2]

        if (
            frame_width != self.config.image_width
            or frame_height != self.config.image_height
        ):
            raise ValueError(
                "Unexpected image size. "
                f"Expected "
                f"{self.config.image_width}x"
                f"{self.config.image_height}, "
                f"received "
                f"{frame_width}x{frame_height}."
            )

        hsv_image = cv2.cvtColor(
            rgb_frame,
            cv2.COLOR_RGB2HSV,
        )

        colour_masks = self._build_colour_masks(
            hsv_image
        )

        observations: list[ObstacleObservation] = []

        for obstacle_name, raw_mask in colour_masks.items():
            mask = self._clean_mask(raw_mask)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                continue

            largest_contour = max(
                contours,
                key=cv2.contourArea,
            )

            contour_area = float(
                cv2.contourArea(largest_contour)
            )

            if (
                contour_area
                < self.config.minimum_contour_area_px
            ):
                continue

            x, y, width, height = cv2.boundingRect(
                largest_contour
            )

            if width < self.config.minimum_width_px:
                continue

            center_x_px = x + width / 2.0
            center_y_px = y + height / 2.0

            bearing_rad = self._estimate_bearing(
                center_x_px
            )

            estimated_distance_m = (
                self._estimate_distance(width)
            )

            bounding_box_area = float(
                width * height
            )

            confidence = self._estimate_confidence(
                contour_area=contour_area,
                bounding_box_area=bounding_box_area,
                width_px=float(width),
            )

            observation = ObstacleObservation(
                obstacle_name=obstacle_name,
                center_x_px=float(center_x_px),
                center_y_px=float(center_y_px),
                width_px=float(width),
                height_px=float(height),
                bearing_rad=bearing_rad,
                estimated_distance_m=estimated_distance_m,
                confidence=confidence,
                timestamp_s=float(timestamp_s),
                bounding_box=(
                    int(x),
                    int(y),
                    int(width),
                    int(height),
                ),
            )

            observations.append(observation)

        observations.sort(
            key=lambda item: item.estimated_distance_m
        )

        return observations

    def draw_detections(
        self,
        rgb_frame: np.ndarray,
        observations: Iterable[ObstacleObservation],
    ) -> np.ndarray:
        """
        Return an annotated BGR frame for OpenCV display.
        """
        bgr_frame = cv2.cvtColor(
            rgb_frame,
            cv2.COLOR_RGB2BGR,
        )

        colour_map = {
            "O1_RED": (0, 0, 255),
            "O2_BLUE": (255, 0, 0),
            "O3_YELLOW": (0, 255, 255),
            "O4_GREEN": (0, 255, 0),
        }

        for observation in observations:
            x, y, width, height = (
                observation.bounding_box
            )

            draw_colour = colour_map.get(
                observation.obstacle_name,
                (255, 255, 255),
            )

            cv2.rectangle(
                bgr_frame,
                (x, y),
                (x + width, y + height),
                draw_colour,
                2,
            )

            center = (
                int(observation.center_x_px),
                int(observation.center_y_px),
            )

            cv2.circle(
                bgr_frame,
                center,
                4,
                draw_colour,
                -1,
            )

            label = (
                f"{observation.obstacle_name} "
                f"d={observation.estimated_distance_m:.2f}m "
                f"b={np.rad2deg(observation.bearing_rad):.1f}deg "
                f"c={observation.confidence:.2f}"
            )

            text_y = max(y - 8, 18)

            cv2.putText(
                bgr_frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                draw_colour,
                1,
                cv2.LINE_AA,
            )

        return bgr_frame