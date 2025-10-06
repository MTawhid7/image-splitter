import cv2
import numpy as np
import logging
from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy


class ProjectionProfileStrategy(BaseSplittingStrategy):
    """
    Splits an image by analyzing projection profiles to find the precise
    bounds of dividers using robust band-scanning.
    """

    def split(self, image: Image) -> SplitResult:
        try:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            center_x = self._find_split_point(np.sum(grayscale_image, axis=0))
            if center_x is None:
                return self._failed_result(
                    "Could not find a candidate for a vertical divider."
                )

            center_y = self._find_split_point(np.sum(grayscale_image, axis=1))
            if center_y is None:
                return self._failed_result(
                    "Could not find a candidate for a horizontal divider."
                )

            # --- THE FINAL, ROBUST UPGRADE: Use Band Scanning ---
            bounds = self._find_bounds_with_band_scan(
                grayscale_image, center_x, center_y
            )
            if bounds is None:
                return self._failed_result(
                    "Failed to determine divider bounds robustly."
                )
            x_start, x_end, y_start, y_end = bounds

            if (x_end - x_start) > image.shape[1] * 0.25 or (
                y_end - y_start
            ) > image.shape[0] * 0.25:
                return self._failed_result(
                    f"Detected divider bounds are unreasonably large: V={x_end-x_start}px, H={y_end-y_start}px."
                )

            logging.debug(f"Found vertical divider bounds: {x_start}-{x_end-1}")
            logging.debug(f"Found horizontal divider bounds: {y_start}-{y_end-1}")

            confidence = self._calculate_confidence(grayscale_image, center_x, center_y)
            if confidence < self.config.get("confidence_threshold", 0.75):
                return self._failed_result(
                    f"Found divider, but confidence {confidence:.2f} is below threshold."
                )

            logging.info(
                f"Found dividers at x=[{x_start}:{x_end}], y=[{y_start}:{y_end}] with confidence {confidence:.2f}"
            )

            height, width, _ = image.shape
            top_left = image[0:y_start, 0:x_start]
            top_right = image[0:y_start, x_end:width]
            bottom_left = image[y_end:height, 0:x_start]
            bottom_right = image[y_end:height, x_end:width]

            return SplitResult(
                success=True,
                strategy_used="projection_profile",
                confidence=confidence,
                images=[top_left, top_right, bottom_left, bottom_right],
            )
        except Exception as e:
            logging.error(f"Error during projection profile splitting: {e}")
            return self._failed_result(str(e))

    def _find_bounds_with_band_scan(
        self, gray_image: Image, center_x: int, center_y: int
    ) -> tuple[int, int, int, int] | None:
        """
        Finds divider bounds by scanning with a band of pixels, using the median
        for robust color sampling. This avoids errors from single-pixel noise or content.
        """
        try:
            height, width = gray_image.shape
            tolerance = self.config.get("divider_color_tolerance", 15)
            band_thickness = 21  # A 21px band is robust. Must be odd.
            half_band = band_thickness // 2

            # Get the robust divider color from the intersection patch
            patch = gray_image[
                max(0, center_y - half_band) : min(height, center_y + half_band + 1),
                max(0, center_x - half_band) : min(width, center_x + half_band + 1),
            ]
            if patch.size == 0:
                return None
            divider_color = float(np.median(patch))

            # --- ROBUST SCANNING for VERTICAL bounds (x_start, x_end) ---
            x_start = center_x
            for x in range(center_x, -1, -1):
                # Sample a vertical band at this x-coordinate
                band = gray_image[
                    max(0, center_y - half_band) : min(
                        height, center_y + half_band + 1
                    ),
                    x,
                ]
                if abs(np.median(band) - divider_color) > tolerance:
                    break
                x_start = x

            x_end = center_x
            for x in range(center_x, width):
                band = gray_image[
                    max(0, center_y - half_band) : min(
                        height, center_y + half_band + 1
                    ),
                    x,
                ]
                if abs(np.median(band) - divider_color) > tolerance:
                    break
                x_end = x

            # --- ROBUST SCANNING for HORIZONTAL bounds (y_start, y_end) ---
            y_start = center_y
            for y in range(center_y, -1, -1):
                # Sample a horizontal band at this y-coordinate
                band = gray_image[
                    y,
                    max(0, center_x - half_band) : min(width, center_x + half_band + 1),
                ]
                if abs(np.median(band) - divider_color) > tolerance:
                    break
                y_start = y

            y_end = center_y
            for y in range(center_y, height):
                band = gray_image[
                    y,
                    max(0, center_x - half_band) : min(width, center_x + half_band + 1),
                ]
                if abs(np.median(band) - divider_color) > tolerance:
                    break
                y_end = y

            return x_start, x_end + 1, y_start, y_end + 1
        except Exception as e:
            logging.error(f"Failed to find divider bounds with band scan: {e}")
            return None

    # ... (The rest of the file: _find_split_point, _calculate_confidence, _failed_result remain unchanged) ...
    def _find_split_point(self, profile: np.ndarray) -> int | None:
        length = len(profile)
        search_ratio = self.config.get("search_zone_ratio", 0.2)
        search_margin = int((length * search_ratio) / 2)
        center = length // 2
        start = center - search_margin
        end = center + search_margin
        search_area = profile[start:end]
        if search_area.size == 0:
            return None
        min_idx, max_idx = np.argmin(search_area), np.argmax(search_area)
        mean_val, min_val, max_val = (
            np.mean(search_area),
            search_area[min_idx],
            search_area[max_idx],
        )
        return (
            int(start + min_idx)
            if (mean_val - min_val) > (max_val - mean_val)
            else int(start + max_idx)
        )

    def _calculate_confidence(
        self, gray_image: Image, split_x: int, split_y: int
    ) -> float:
        consistency_threshold = self.config.get("consistency_threshold", 10.0)
        std_dev_v = np.std(gray_image[:, split_x])
        std_dev_h = np.std(gray_image[split_y, :])
        v_consistency = max(0, 1.0 - (std_dev_v / consistency_threshold))
        h_consistency = max(0, 1.0 - (std_dev_h / consistency_threshold))
        return (v_consistency + h_consistency) / 2.0

    def _failed_result(self, message: str) -> SplitResult:
        logging.warning(f"Projection Profile Strategy failed: {message}")
        return SplitResult(
            success=False,
            strategy_used="projection_profile",
            confidence=0.0,
            error_message=message,
        )
