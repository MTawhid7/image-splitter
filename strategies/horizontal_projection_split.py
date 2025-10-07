import cv2
import numpy as np
import logging
from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy
from utils.image_utils import find_precise_bounds


class HorizontalProjectionSplitStrategy(BaseSplittingStrategy):
    """
    A hybrid strategy for images with only a horizontal divider.
    It uses projection profile to find the horizontal split and a simple
    midpoint for the vertical split.
    """

    def split(self, image: Image) -> SplitResult:
        try:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width, _ = image.shape

            # 1. Use Projection Profile for the horizontal axis
            profile = np.sum(grayscale, axis=1)
            center_y = int(
                np.argmin(profile[int(height * 0.4) : int(height * 0.6)])
                + int(height * 0.4)
            )
            bounds = find_precise_bounds(grayscale, width // 2, center_y, self.config)
            if bounds is None:
                return self._failed_result("Could not find horizontal bounds.")
            _, _, y_start, y_end = bounds

            # 2. Use a simple midpoint for the vertical axis
            x_split = width // 2

            images = [
                image[0:y_start, 0:x_split],
                image[0:y_start, x_split:width],
                image[y_end:height, 0:x_split],
                image[y_end:height, x_split:width],
            ]
            return SplitResult(
                success=True,
                strategy_used="horizontal_projection_split",
                confidence=0.9,
                images=images,
            )
        except Exception as e:
            return self._failed_result(str(e))

    def _failed_result(self, msg):
        logging.warning(f"Horizontal Projection Split failed: {msg}")
        return SplitResult(
            success=False, strategy_used="horizontal_projection_split", confidence=0.0
        )
