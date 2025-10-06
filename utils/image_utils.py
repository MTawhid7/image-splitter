import cv2
import logging
import os
from typing import Optional
import numpy as np

# Define a type alias for images for clarity
Image = np.ndarray


def load_image(image_path: str) -> Optional[Image]:
    """
    Loads an image from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Optional[Image]: The loaded image as a NumPy array, or None if loading fails.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file not found at: {image_path}")
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(
                f"Failed to read image from {image_path}. It may be corrupt or an unsupported format."
            )
            return None
        logging.debug(
            f"Successfully loaded image: {image_path} with dimensions {image.shape}"
        )
        return image
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {image_path}: {e}")
        return None


def save_image(image: Image, output_path: str) -> bool:
    """
    Saves an image to the specified path.

    Args:
        image (Image): The image to save (as a NumPy array).
        output_path (str): The path where the image will be saved.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        # Create the directory if it doesn't exist
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
