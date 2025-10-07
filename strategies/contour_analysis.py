import cv2
import numpy as np
import logging
from typing import List, Tuple

from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy
from utils.debug_utils import save_contour_debug_image


class ContourAnalysisStrategy(BaseSplittingStrategy):
    """
    A robust strategy for seamless images that uses a corner-based sorting
    algorithm to correctly order panels regardless of their alignment.
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

            if len(valid_contours) < 4:
                return self._failed_result(
                    f"Found only {len(valid_contours)} distinct content areas. Needed 4."
                )

            valid_contours.sort(key=cv2.contourArea, reverse=True)
            top_4_contours = valid_contours[:4]

            bounding_boxes: List[Tuple[int, int, int, int]] = []
            for c in top_4_contours:
                x, y, w, h = cv2.boundingRect(c)
                bounding_boxes.append((int(x), int(y), int(w), int(h)))

            # --- THE ROBUST FIX: Use corner-based sorting instead of fragile row sorting ---
            # Calculate the center of each bounding box
            centers = [
                (int(x + w / 2), int(y + h / 2)) for x, y, w, h in bounding_boxes
            ]

            # Sort boxes based on their center's distance to each corner of the image
            # Top-Left is closest to (0, 0)
            tl_box = min(
                bounding_boxes,
                key=lambda b: np.sqrt((b[0] + b[2] / 2) ** 2 + (b[1] + b[3] / 2) ** 2),
            )
            # Top-Right is closest to (width, 0)
            tr_box = min(
                bounding_boxes,
                key=lambda b: np.sqrt(
                    (b[0] + b[2] / 2 - width) ** 2 + (b[1] + b[3] / 2) ** 2
                ),
            )
            # Bottom-Left is closest to (0, height)
            bl_box = min(
                bounding_boxes,
                key=lambda b: np.sqrt(
                    (b[0] + b[2] / 2) ** 2 + (b[1] + b[3] / 2 - height) ** 2
                ),
            )
            # Bottom-Right is closest to (width, height)
            br_box = min(
                bounding_boxes,
                key=lambda b: np.sqrt(
                    (b[0] + b[2] / 2 - width) ** 2 + (b[1] + b[3] / 2 - height) ** 2
                ),
            )

            sorted_boxes = [tl_box, tr_box, bl_box, br_box]

            cropped_images = [
                image[y : y + h, x : x + w] for (x, y, w, h) in sorted_boxes
            ]
            relative_bounds = [(0, 0, int(w), int(h)) for _, _, w, h in sorted_boxes]

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

            return SplitResult(
                success=True,
                strategy_used="contour_analysis",
                confidence=0.9,
                images=cropped_images,
                bounds=relative_bounds,
            )
        except Exception as e:
            return self._failed_result(str(e))

    def _failed_result(self, message: str) -> SplitResult:
        logging.warning(f"Contour Analysis failed: {message}")
        return SplitResult(
            success=False,
            strategy_used="contour_analysis",
            confidence=0.0,
            error_message=message,
        )
