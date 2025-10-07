import cv2
import numpy as np
import logging
from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy
from utils.image_utils import find_precise_bounds


class VerticalProjectionSplitStrategy(BaseSplittingStrategy):
    """
    A hybrid strategy for images with only a vertical divider.
    It uses projection profile to find the vertical split and a simple
    midpoint for the horizontal split.
    """

    def split(self, image: Image) -> SplitResult:
        try:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width, _ = image.shape

            # 1. Use Projection Profile for the vertical axis
            profile = np.sum(grayscale, axis=0)
            center_x = int(
                np.argmin(profile[int(width * 0.4) : int(width * 0.6)])
                + int(width * 0.4)
            )
            bounds = find_precise_bounds(grayscale, center_x, height // 2, self.config)
            if bounds is None:
                return self._failed_result("Could not find vertical bounds.")
            x_start, x_end, _, _ = bounds

            # 2. Use a simple midpoint for the horizontal axis
            y_split = height // 2

            images = [
                image[0:y_split, 0:x_start],
                image[0:y_split, x_end:width],
                image[y_split:height, 0:x_start],
                image[y_split:height, x_end:width],
            ]
            return SplitResult(
                success=True,
                strategy_used="vertical_projection_split",
                confidence=0.9,
                images=images,
            )
        except Exception as e:
            return self._failed_result(str(e))

    def _failed_result(self, msg):
        logging.warning(f"Vertical Projection Split failed: {msg}")
        return SplitResult(
            success=False, strategy_used="vertical_projection_split", confidence=0.0
        )
