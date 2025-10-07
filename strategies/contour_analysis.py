import cv2
import numpy as np
import logging
from typing import List, Tuple

from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy
from utils.debug_utils import save_contour_debug_image


class ContourAnalysisStrategy(BaseSplittingStrategy):
    """
    A robust strategy for seamless images. It uses adaptive thresholding and
    morphological operations to isolate content 'blobs', then identifies
    and crops the four largest distinct areas.
    """

    def split(self, image: Image, filename: str) -> SplitResult:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            total_area = height * width

            cfg = self.config
            block_size = cfg.get("adaptive_block_size", 15)
            c_value = cfg.get("adaptive_c_value", 4)
            kernel_size = cfg.get("morph_kernel_size", 5)
            min_area_ratio = cfg.get("min_contour_area_ratio", 0.01)
            morph_operation = cfg.get("morph_operation", "close")

            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                c_value,
            )

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            cleaned_mask = (
                cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                if morph_operation == "close"
                else cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            )

            contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            min_area = total_area * min_area_ratio
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

            logging.debug(
                f"Contour Analysis: Found {len(contours)} total, {len(valid_contours)} valid contours."
            )

            if len(valid_contours) < 4:
                return self._failed_result(
                    f"Found only {len(valid_contours)} distinct content areas. Needed 4."
                )

            valid_contours.sort(key=cv2.contourArea, reverse=True)
            top_4_contours = valid_contours[:4]

            # cv2.boundingRect returns numpy integers; explicitly cast to Python ints
            # to satisfy the strict type checking required by downstream functions.
            bounding_boxes: List[Tuple[int, int, int, int]] = []
            for c in top_4_contours:
                x, y, w, h = cv2.boundingRect(c)
                bounding_boxes.append((int(x), int(y), int(w), int(h)))

            y_sorted = sorted(bounding_boxes, key=lambda b: b[1])
            mid_y_split = (y_sorted[1][1] + y_sorted[2][1]) / 2
            top_row = [b for b in bounding_boxes if b[1] < mid_y_split]
            bottom_row = [b for b in bounding_boxes if b[1] >= mid_y_split]

            if len(top_row) != 2 or len(bottom_row) != 2:
                return self._failed_result(
                    "Could not spatially arrange contours into a 2x2 grid."
                )

            top_row.sort(key=lambda b: b[0])
            bottom_row.sort(key=lambda b: b[0])
            sorted_boxes = top_row + bottom_row

            cropped_images = [
                image[y : y + h, x : x + w] for (x, y, w, h) in sorted_boxes
            ]

            # --- LOGIC FIX: The content is the *entire* cropped panel. ---
            # The bounds relative to the panel are (0, 0, width, height).
            # Explicitly cast to int to satisfy the strict type checking.
            relative_bounds: List[Tuple[int, int, int, int]] = [
                (0, 0, int(w), int(h)) for _, _, w, h in sorted_boxes
            ]

            if self.debug:
                debug_dir = "output_results/debug"
                save_contour_debug_image(
                    image,
                    valid_contours,
                    top_4_contours,
                    sorted_boxes,
                    debug_dir,
                    filename,
                )

            logging.info(
                "Successfully isolated and sorted 4 content panels using Contour Analysis."
            )
            return SplitResult(
                success=True,
                strategy_used="contour_analysis",
                confidence=0.9,
                images=cropped_images,
                bounds=relative_bounds,
            )

        except Exception as e:
            logging.error(f"Error in ContourAnalysisStrategy: {e}", exc_info=True)
            return self._failed_result(str(e))

    def _failed_result(self, message: str) -> SplitResult:
        logging.warning(f"Contour Analysis failed: {message}")
        return SplitResult(
            success=False,
            strategy_used="contour_analysis",
            confidence=0.0,
            error_message=message,
        )
