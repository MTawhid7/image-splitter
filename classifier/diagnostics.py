from enum import Enum, auto
import cv2
import numpy as np
from core.data_models import Image


class ImageType(Enum):
    """Defines the possible classifications for a composite image."""

    DIVIDERS_FULL = auto()
    DIVIDERS_VERTICAL_ONLY = auto()
    DIVIDERS_HORIZONTAL_ONLY = auto()
    SEAMLESS_UNIFORM = auto()
    SEAMLESS_COMPLEX = auto()
    UNKNOWN = auto()


def has_full_dividers(image: Image, config: dict) -> bool:
    """
    Checks for the presence of strong horizontal and vertical divider lines
    using the Hough Line Transform.
    """
    cfg = config.get("classifier", {})

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=cfg.get("hough_threshold", 100),
        minLineLength=cfg.get("hough_min_line_length", 150),
        maxLineGap=cfg.get("hough_max_line_gap", 20),
    )

    if lines is None:
        return False

    h_lines, v_lines = 0, 0
    height, width, _ = image.shape

    center_margin_x = int(width * 0.2)
    center_margin_y = int(height * 0.2)

    # --- THE FIX IS HERE ---
    # Iterate through the array of lines more explicitly.
    for i in range(lines.shape[0]):
        # Each 'line' is an array within the main array, e.g., [[x1, y1, x2, y2]]
        line = lines[i][0]
        x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])

        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))

        if abs(angle) < 5 and (center_margin_y < y1 < height - center_margin_y):
            h_lines += 1
        elif abs(abs(angle) - 90) < 5 and (
            center_margin_x < x1 < width - center_margin_x
        ):
            v_lines += 1

    found_h = h_lines >= cfg.get("min_h_lines", 1)
    found_v = v_lines >= cfg.get("min_v_lines", 1)

    return found_h and found_v
