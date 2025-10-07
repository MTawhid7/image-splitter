import logging
import cv2
import numpy as np
from core.data_models import Image
from classifier.diagnostics import ImageType


class ImageClassifier:
    """
    Analyzes an image to determine its structural type using a robust,
    multi-stage diagnostic process.
    """

    def __init__(self, config: dict):
        self.config = config.get("classifier", {})

    def _has_divider(self, gray_image: np.ndarray, axis: int) -> bool:
        """
        A divider is a line of low variance in a region of high average variance.
        """
        h, w = gray_image.shape
        search_ratio = self.config.get(
            "search_zone_ratio", 0.3
        )  # Widen search for robustness
        variance_threshold = self.config.get("divider_variance_threshold", 50)
        # A divider is only valid if the surrounding area is at least this much noisier
        mean_multiplier = self.config.get("mean_variance_multiplier", 1.5)

        if axis == 0:  # Horizontal divider
            center, margin = h // 2, int(h * search_ratio / 2)
            search_area = gray_image[center - margin : center + margin, :]
            variances = np.std(search_area, axis=1)
        else:  # Vertical divider
            center, margin = w // 2, int(w * search_ratio / 2)
            search_area = gray_image[:, center - margin : center + margin]
            variances = np.std(search_area, axis=0)

        if variances.size == 0:
            return False

        min_variance = np.min(variances)
        mean_variance = np.mean(variances)

        # Condition 1: The divider line itself must be very uniform (low variance)
        is_uniform_line = min_variance < variance_threshold
        # Condition 2: The surrounding area must be significantly more varied (i.e., contain content)
        is_distinct_from_surroundings = mean_variance > min_variance * mean_multiplier

        return is_uniform_line and is_distinct_from_surroundings

    def _has_uniform_background(self, image: Image) -> bool:
        h, w, _ = image.shape
        margin = 15
        pixels = np.concatenate(
            [
                image[0:margin, :].reshape(-1, 3),
                image[h - margin : h, :].reshape(-1, 3),
                image[:, 0:margin].reshape(-1, 3),
                image[:, w - margin : w].reshape(-1, 3),
            ]
        )
        return np.mean(np.std(pixels, axis=0)) < 15

    def diagnose(self, image: Image) -> ImageType:
        logging.info("Classifier: Diagnosing image type...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        has_h_divider = self._has_divider(gray, axis=0)
        has_v_divider = self._has_divider(gray, axis=1)

        if has_h_divider and has_v_divider:
            logging.info("Classifier: Diagnosis is DIVIDERS_FULL.")
            return ImageType.DIVIDERS_FULL
        if has_h_divider:
            logging.info("Classifier: Diagnosis is DIVIDERS_HORIZONTAL_ONLY.")
            return ImageType.DIVIDERS_HORIZONTAL_ONLY
        if has_v_divider:
            logging.info("Classifier: Diagnosis is DIVIDERS_VERTICAL_ONLY.")
            return ImageType.DIVIDERS_VERTICAL_ONLY

        logging.info("Classifier: No robust dividers found. Analyzing background...")
        if self._has_uniform_background(image):
            logging.info("Classifier: Diagnosis is SEAMLESS_UNIFORM.")
            return ImageType.SEAMLESS_UNIFORM
        else:
            logging.info("Classifier: Diagnosis is SEAMLESS_COMPLEX.")
            return ImageType.SEAMLESS_COMPLEX
