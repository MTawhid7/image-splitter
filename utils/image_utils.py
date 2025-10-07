import cv2
import logging
import os
from typing import Optional
import numpy as np

Image = np.ndarray

def load_image(image_path: str) -> Optional[Image]:
    """Loads an image from the specified path."""
    if not os.path.exists(image_path):
        logging.error(f"Image file not found at: {image_path}")
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image from {image_path}. It may be corrupt or an unsupported format.")
            return None
        logging.debug(f"Successfully loaded image: {image_path} with dimensions {image.shape}")
        return image
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {image_path}: {e}")
        return None

def save_image(image: Image, output_path: str) -> bool:
    """Saves an image to the specified path."""
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.debug(f"Created output directory: {output_dir}")
        cv2.imwrite(output_path, image)
        logging.debug(f"Successfully saved image to: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save image to {output_path}: {e}")
        return False


def find_content_bounds(image: Image, config: dict) -> tuple[int, int, int, int] | None:
    """
    Finds the bounding box of the main content in an image panel using a robust
    edge-detection (Canny) method.
    """
    # --- ROBUSTNESS FIX: Check if the image is empty ---
    if image is None or image.size == 0:
        logging.warning("Trim: Received an empty image. Cannot find content.")
        return None
    try:
        trim_config = config.get("trimming", {})
        canny_threshold = trim_config.get("canny_threshold", 30)

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_threshold, canny_threshold * 2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        all_points = np.vstack([c for c in contours])
        x, y, w, h = cv2.boundingRect(all_points)
        return (int(x), int(y), int(w), int(h))
    except Exception as e:
        logging.error(f"Error during edge-based content bounds detection: {e}.")
        return None


def find_precise_bounds(gray_image: Image, center_x: int, center_y: int, config: dict) -> tuple[int, int, int, int] | None:
    """Finds precise divider bounds using robust band-scanning."""
    try:
        height, width = gray_image.shape
        tolerance = config.get('divider_color_tolerance', 15)
        band_thickness = config.get('band_thickness', 21)
        half_band = band_thickness // 2
        patch = gray_image[max(0, center_y - half_band):min(height, center_y + half_band + 1),
                           max(0, center_x - half_band):min(width, center_x + half_band + 1)]
        if patch.size == 0: return None
        divider_color = float(np.median(patch))
        x_start = center_x
        for x in range(center_x, -1, -1):
            band = gray_image[max(0, center_y - half_band):min(height, center_y + half_band + 1), x]
            if abs(np.median(band) - divider_color) > tolerance: break
            x_start = x
        x_end = center_x
        for x in range(center_x, width):
            band = gray_image[max(0, center_y - half_band):min(height, center_y + half_band + 1), x]
            if abs(np.median(band) - divider_color) > tolerance: break
            x_end = x
        y_start = center_y
        for y in range(center_y, -1, -1):
            band = gray_image[y, max(0, center_x - half_band):min(width, center_x + half_band + 1)]
            if abs(np.median(band) - divider_color) > tolerance: break
            y_start = y
        y_end = center_y
        for y in range(center_y, height):
            band = gray_image[y, max(0, center_x - half_band):min(width, center_x + half_band + 1)]
            if abs(np.median(band) - divider_color) > tolerance: break
            y_end = y
        return x_start, x_end + 1, y_start, y_end + 1
    except Exception as e:
        logging.error(f"Failed to find precise bounds with band scan: {e}")
        return None
