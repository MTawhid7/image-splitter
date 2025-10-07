import logging
import cv2
import numpy as np
from core.data_models import Image
from classifier.diagnostics import ImageType


class ImageClassifier:
    """
    Analyzes an image to determine its structural type using a robust,
    multi-stage diagnostic process based on background connectivity.
    """

    def __init__(self, config: dict):
        self.config = config.get("classifier", {})

    def _get_background_mask(self, image: Image) -> np.ndarray:
        """
        Identifies the dominant background color from the corners and returns
        a binary mask of that background.
        """
        h, w, _ = image.shape
        margin = 15
        # Sample all four corners to get a robust median background color
        corners = np.array(
            [
                image[margin, margin],
                image[margin, w - margin],
                image[h - margin, margin],
                image[h - margin, w - margin],
            ]
        )
        bg_color = np.median(corners, axis=0)

        # Define a color tolerance range
        tolerance = self.config.get("background_color_tolerance", 25)
        lower_bound = np.maximum(0, bg_color - tolerance).astype(int)
        upper_bound = np.minimum(255, bg_color + tolerance).astype(int)

        # Create the binary mask
        mask = cv2.inRange(image, lower_bound, upper_bound)
        return mask

    def _diagnose_structure(self, image: Image) -> ImageType:
        """
        The core of the new classifier. It analyzes the connectivity of the
        background to determine if dividers are present.
        """
        bg_mask = self._get_background_mask(image)

        # 1. Clean up small noise in the background mask
        closing_kernel_size = self.config.get("closing_kernel_size", 3)
        closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
        cleaned_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, closing_kernel)

        # 2. Erode the mask to break the thin divider connections
        erosion_kernel_size = self.config.get("erosion_kernel_size", 15)
        erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        eroded_mask = cv2.erode(cleaned_mask, erosion_kernel, iterations=1)

        # 3. Count the number of resulting distinct background blobs
        contours, _ = cv2.findContours(
            eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter out tiny noise contours
        min_area = eroded_mask.shape[0] * eroded_mask.shape[1] * 0.01
        large_blobs = [c for c in contours if cv2.contourArea(c) > min_area]
        blob_count = len(large_blobs)
        logging.debug(
            f"Classifier: Found {blob_count} distinct background blobs after erosion."
        )

        # 4. Make the diagnosis based on the blob count
        if blob_count >= 4:
            logging.info(
                "Classifier: Diagnosis is DIVIDERS_FULL (background is disconnected)."
            )
            return ImageType.DIVIDERS_FULL
        else:
            # If the background is still one piece, it's seamless.
            # Now we can use a simple check to see if it's uniform or complex.
            if (
                np.mean(np.std(image, axis=(0, 1))) < 20
            ):  # Check overall image color variance
                logging.info(
                    "Classifier: Diagnosis is SEAMLESS_UNIFORM (background is connected)."
                )
                return ImageType.SEAMLESS_UNIFORM
            else:
                logging.info(
                    "Classifier: Diagnosis is SEAMLESS_COMPLEX (background is connected)."
                )
                return ImageType.SEAMLESS_COMPLEX

    def diagnose(self, image: Image) -> ImageType:
        """
        Public method to run the diagnostic pipeline.
        """
        # This new method is robust enough to be the only check we need.
        return self._diagnose_structure(image)
