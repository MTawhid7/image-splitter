import cv2
import numpy as np
import logging
from typing import List, Tuple

from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy
from utils.image_utils import find_precise_bounds, find_content_bounds
from utils.debug_utils import save_projection_debug_image


class ProjectionProfileStrategy(BaseSplittingStrategy):
    """
    Finds dividers by identifying the path of lowest standard deviation and
    returns panels with their internal content bounds.
    """

    def split(self, image: Image, filename: str) -> SplitResult:
        try:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = grayscale_image.shape

            center_x = self._find_split_point_by_variance(grayscale_image, axis=1)
            center_y = self._find_split_point_by_variance(grayscale_image, axis=0)

            if center_x is None or center_y is None:
                return self._failed_result(
                    "Could not find a valid low-variance split point."
                )

            bounds = find_precise_bounds(
                grayscale_image, center_x, center_y, self.config
            )
            if bounds is None:
                return self._failed_result(
                    "Failed to determine precise divider bounds."
                )
            x_start, x_end, y_start, y_end = bounds

            confidence = self._calculate_confidence(grayscale_image, center_x, center_y)
            if confidence < self.config.get("confidence_threshold", 0.75):
                return self._failed_result(
                    f"Divider confidence {confidence:.2f} is below threshold."
                )

            logging.info(
                f"Found dividers at x=[{x_start}:{x_end}], y=[{y_start}:{y_end}] with confidence {confidence:.2f}"
            )

            if self.debug:
                debug_dir = "output_results/debug"
                save_projection_debug_image(
                    image,
                    center_x,
                    center_y,
                    bounds,
                    debug_dir,
                    filename,
                )

            images = [
                image[0:y_start, 0:x_start],
                image[0:y_start, x_end:width],
                image[y_end:height, 0:x_start],
                image[y_end:height, x_end:width],
            ]

            # --- FIX 1: Explicitly build the list to satisfy the type checker ---
            relative_bounds: List[Tuple[int, int, int, int]] = []
            for panel in images:
                content_bound = find_content_bounds(panel, self.config)
                if content_bound:
                    # Ensure a fixed-size 4-tuple for the type checker by indexing explicitly
                    cb0 = int(content_bound[0])
                    cb1 = int(content_bound[1])
                    cb2 = int(content_bound[2])
                    cb3 = int(content_bound[3])
                    relative_bounds.append((cb0, cb1, cb2, cb3))
                else:
                    relative_bounds.append((0, 0, 0, 0))

            return SplitResult(
                success=True,
                strategy_used="projection_profile",
                confidence=confidence,
                images=images,
                bounds=relative_bounds,
            )
        except Exception as e:
            return self._failed_result(str(e))

    def _find_split_point_by_variance(self, gray_image: Image, axis: int) -> int | None:
        """Finds the index with the minimum standard deviation in the central search zone."""
        h, w = gray_image.shape
        search_ratio = self.config.get("search_zone_ratio", 0.2)

        if axis == 0:  # Horizontal divider
            center = h // 2
            margin = int(h * search_ratio / 2)
            start, end = center - margin, center + margin
            search_area = gray_image[start:end, :]
            variances = np.std(search_area, axis=1)
        else:  # Vertical divider
            center = w // 2
            margin = int(w * search_ratio / 2)
            start, end = center - margin, center + margin
            search_area = gray_image[:, start:end]
            variances = np.std(search_area, axis=0)

        return int(start + np.argmin(variances)) if variances.size > 0 else None

    def _calculate_confidence(
        self, gray_image: Image, split_x: int, split_y: int
    ) -> float:
        """Confidence is high when the standard deviation of the divider path is low."""
        max_variance = 50.0
        std_v = np.std(gray_image[:, split_x])
        std_h = np.std(gray_image[split_y, :])

        conf_v = max(0.0, 1.0 - (std_v / max_variance))
        conf_h = max(0.0, 1.0 - (std_h / max_variance))

        # --- FIX 2: Explicitly cast the final result to a Python float ---
        return float((conf_v + conf_h) / 2.0)

    def _failed_result(self, message: str) -> SplitResult:
        logging.warning(f"Projection Profile Strategy failed: {message}")
        return SplitResult(
            success=False,
            strategy_used="projection_profile",
            confidence=0.0,
            error_message=message,
        )
