import cv2
import numpy as np
import logging
from core.data_models import SplitResult, Image
from strategies.base_strategy import BaseSplittingStrategy


class ContourAnalysisStrategy(BaseSplittingStrategy):
    """
    Splits an image by finding all content contours, sorting them into quadrants,
    and creating a master bounding box for each quadrant's content.
    """

    def split(self, image: Image) -> SplitResult:
        try:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

            canny_thresh1 = self.config.get("canny_threshold1", 50)
            canny_thresh2 = self.config.get("canny_threshold2", 150)
            edges = cv2.Canny(blurred_image, canny_thresh1, canny_thresh2)

            # --- UPGRADE: Find ALL contours, not just external ones ---
            contours, _ = cv2.findContours(
                edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return self._failed_result("No contours found in the image.")

            # --- NEW LOGIC: Sort contours into quadrant buckets ---
            height, width = image.shape[:2]
            mid_x, mid_y = width // 2, height // 2

            quadrant_contours = {"tl": [], "tr": [], "bl": [], "br": []}

            for c in contours:
                # Filter out tiny noise contours
                if cv2.contourArea(c) < 100:
                    continue

                # Get the center of the contour
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Assign to a bucket based on its center
                if cx < mid_x and cy < mid_y:
                    quadrant_contours["tl"].append(c)
                elif cx >= mid_x and cy < mid_y:
                    quadrant_contours["tr"].append(c)
                elif cx < mid_x and cy >= mid_y:
                    quadrant_contours["bl"].append(c)
                else:
                    quadrant_contours["br"].append(c)

            # --- NEW LOGIC: Create a master bounding box for each bucket ---
            final_boxes = []
            for quad in ["tl", "tr", "bl", "br"]:
                if not quadrant_contours[quad]:
                    return self._failed_result(
                        f"No content contours found in the {quad} quadrant."
                    )

                # Combine all contours in the quadrant into one mega-contour
                all_points = np.vstack(quadrant_contours[quad])
                # Get the bounding box of the combined points
                final_boxes.append(cv2.boundingRect(all_points))

            confidence = self._validate_grid_layout(final_boxes)
            if confidence < self.config.get("confidence_threshold", 0.75):
                return self._failed_result(
                    f"Panel layout validation failed. Confidence {confidence:.2f} is below threshold."
                )

            images = [image[y : y + h, x : x + w] for (x, y, w, h) in final_boxes]

            return SplitResult(
                success=True,
                strategy_used="contour_analysis",
                confidence=confidence,
                images=images,
            )

        except Exception as e:
            logging.error(f"Error during contour analysis splitting: {e}")
            return self._failed_result(str(e))

    # ... (The _validate_grid_layout and _failed_result methods remain unchanged) ...
    def _validate_grid_layout(self, boxes: list) -> float:
        if len(boxes) != 4:
            return 0.0
        areas = [w * h for x, y, w, h in boxes]
        mean_area = np.mean(areas)
        if mean_area == 0:
            return 0.0
        std_dev_area = np.std(areas)
        area_similarity_score = max(0, 1.0 - (std_dev_area / mean_area))
        area_threshold = self.config.get("panel_area_similarity_threshold", 0.2)
        if (std_dev_area / mean_area) > area_threshold:
            logging.warning(
                f"Panel areas are not similar enough. CoV: {std_dev_area / mean_area:.2f}"
            )
            return 0.0
        centroids = [(x + w // 2, y + h // 2) for x, y, w, h in boxes]
        centroids.sort(key=lambda p: (p[1], p[0]))
        tl, tr, bl, br = centroids
        tolerance = self.config.get("grid_alignment_tolerance_px", 50)
        align_y1 = abs(tl[1] - tr[1]) < tolerance
        align_y2 = abs(bl[1] - br[1]) < tolerance
        align_x1 = abs(tl[0] - bl[0]) < tolerance
        align_x2 = abs(tr[0] - br[0]) < tolerance
        grid_score = sum([align_y1, align_y2, align_x1, align_x2]) / 4.0
        confidence = (area_similarity_score * 0.5) + (grid_score * 0.5)
        return float(confidence)

    def _failed_result(self, message: str) -> SplitResult:
        logging.warning(f"Contour Analysis Strategy failed: {message}")
        return SplitResult(
            success=False,
            strategy_used="contour_analysis",
            confidence=0.0,
            error_message=message,
        )
