import logging
from typing import List
from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy


class MidpointFallbackStrategy(BaseSplittingStrategy):
    """
    A simple fallback strategy that splits the image into four equal quadrants.
    """

    # --- FIX: Add 'filename' to the method signature to match the base class ---
    def split(self, image: Image, filename: str) -> SplitResult:
        try:
            height, width, _ = image.shape
            mid_x, mid_y = width // 2, height // 2

            top_left = image[0:mid_y, 0:mid_x]
            top_right = image[0:mid_y, mid_x:width]
            bottom_left = image[mid_y:height, 0:mid_x]
            bottom_right = image[mid_y:height, mid_x:width]

            split_images: List[Image] = [top_left, top_right, bottom_left, bottom_right]

            # This strategy does not find content, so it returns zero-size bounds
            bounds = [(0, 0, 0, 0)] * 4

            return SplitResult(
                success=True,
                strategy_used="midpoint_fallback",
                confidence=0.2,
                images=split_images,
                bounds=bounds,
            )
        except Exception as e:
            return SplitResult(
                success=False,
                strategy_used="midpoint_fallback",
                confidence=0.0,
                error_message=str(e),
            )
