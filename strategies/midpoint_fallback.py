import logging
from typing import List
from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy


class MidpointFallbackStrategy(BaseSplittingStrategy):
    """
    A simple fallback strategy that splits the image into four equal quadrants
    based on its midpoint. It does not perform any image analysis.
    """

    def split(self, image: Image) -> SplitResult:
        """
        Splits the image at its horizontal and vertical center.

        Args:
            image (Image): The input image to be split.

        Returns:
            SplitResult: An object containing the four quadrants and a fixed confidence score.
        """
        try:
            height, width, _ = image.shape
            mid_x = width // 2
            mid_y = height // 2

            logging.debug(
                f"Splitting image of size {width}x{height} at midpoint ({mid_x}, {mid_y})"
            )

            # Crop the image into four quadrants using array slicing
            top_left = image[0:mid_y, 0:mid_x]
            top_right = image[0:mid_y, mid_x:width]
            bottom_left = image[mid_y:height, 0:mid_x]
            bottom_right = image[mid_y:height, mid_x:width]

            split_images: List[Image] = [top_left, top_right, bottom_left, bottom_right]

            return SplitResult(
                success=True,
                strategy_used="midpoint_fallback",
                # This strategy is always "confident" in its result, but we assign
                # it a low score so that more intelligent strategies can override it.
                confidence=0.2,  # A low but non-zero confidence
                images=split_images,
            )
        except Exception as e:
            logging.error(f"Error during midpoint splitting: {e}")
            return SplitResult(
                success=False,
                strategy_used="midpoint_fallback",
                confidence=0.0,
                error_message=str(e),
            )
