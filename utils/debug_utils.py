import cv2
import numpy as np
import os
from typing import List, Tuple

Image = np.ndarray


def save_projection_debug_image(
    original_image: Image,
    split_x: int,
    split_y: int,
    bounds: Tuple[int, int, int, int],
    output_dir: str,
    filename: str,  # Now used to create a unique name
):
    debug_image = original_image.copy()
    h, w, _ = debug_image.shape
    x_start, x_end, y_start, y_end = bounds

    cv2.line(debug_image, (split_x, 0), (split_x, h), (255, 0, 0), 2)
    cv2.line(debug_image, (0, split_y), (w, split_y), (255, 0, 0), 2)
    cv2.rectangle(debug_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- FIX: Make the output filename dynamic ---
    base_name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{base_name}_debug_projection{ext}")
    cv2.imwrite(output_path, debug_image)


def save_contour_debug_image(
    original_image: Image,
    contours: List[np.ndarray],
    selected_contours: List[np.ndarray],
    sorted_boxes: List[Tuple[int, int, int, int]],
    output_dir: str,
    filename: str,  # Now used to create a unique name
):
    debug_image = original_image.copy()
    cv2.drawContours(debug_image, contours, -1, (255, 0, 0), 2)
    cv2.drawContours(debug_image, selected_contours, -1, (0, 255, 0), 3)
    labels = ["1_TL", "2_TR", "3_BL", "4_BR"]
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            debug_image,
            labels[i],
            (x + 5, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- FIX: Make the output filename dynamic ---
    base_name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{base_name}_debug_contours{ext}")
    cv2.imwrite(output_path, debug_image)
