import logging
import os
from typing import List, Tuple, Optional
import numpy as np

from classifier.diagnostics import ImageType
from classifier.image_classifier import ImageClassifier
from core.data_models import Image, SplitResult
from core.image_splitter import ImageSplitter
from utils.image_utils import save_image, find_content_bounds


# The standardize_and_center_panels function is logically sound for its specific purpose.
# No changes are needed here.
def standardize_and_center_panels(
    panels: List[Image], main_bg_color: List[int], padding: int, config: dict
) -> List[Image]:
    subject_bounds: List[Optional[Tuple[int, int, int, int]]] = []
    for panel in panels:
        subject_bounds.append(find_content_bounds(panel, config))

    valid_bounds = [
        b for b in subject_bounds if b is not None and b[2] > 0 and b[3] > 0
    ]
    if not valid_bounds:
        logging.warning("No valid subjects found in panels for standardization.")
        canvas_w, canvas_h = 100, 100
        return [np.full((canvas_h, canvas_w, 3), main_bg_color, dtype=np.uint8)] * len(
            panels
        )

    max_w = max(b[2] for b in valid_bounds)
    max_h = max(b[3] for b in valid_bounds)
    canvas_w, canvas_h = max_w + (padding * 2), max_h + (padding * 2)
    logging.info(f"Standardizing all subjects to a canvas of {canvas_w}x{canvas_h}")

    final_panels = []
    for i, panel in enumerate(panels):
        canvas = np.full((canvas_h, canvas_w, 3), main_bg_color, dtype=np.uint8)
        current_subject_bounds = subject_bounds[i]
        if current_subject_bounds:
            x, y, w, h = current_subject_bounds
            subject = panel[y : y + h, x : x + w]
            paste_x, paste_y = (canvas_w - w) // 2, (canvas_h - h) // 2
            canvas[paste_y : paste_y + h, paste_x : paste_x + w] = subject
        final_panels.append(canvas)
    return final_panels


def process_image(
    image: Image, filename: str, splitter: ImageSplitter, config: dict, output_dir: str
):
    trim_config = config.get("trimming", {})
    trim_enabled = trim_config.get("enabled", False)
    padding = trim_config.get("padding", 15)

    h, w, _ = image.shape
    margin = 15
    corners = np.array(
        [
            image[margin, margin],
            image[margin, w - margin],
            image[h - margin, margin],
            image[h - margin, w - margin],
        ]
    )
    main_bg_color = np.median(corners, axis=0).astype(int).tolist()

    classifier = ImageClassifier(config)
    image_type = classifier.diagnose(image)

    strategy_name = None
    if image_type == ImageType.DIVIDERS_FULL:
        strategy_name = "projection_profile"
    elif image_type in [ImageType.SEAMLESS_UNIFORM, ImageType.SEAMLESS_COMPLEX]:
        strategy_name = "contour_analysis"

    result: SplitResult | None = None
    if strategy_name:
        strategy = splitter.get_strategy(strategy_name)
        if strategy:
            result = strategy.split(image, filename)

    if not (result and result.success):
        result = splitter.run_full_pipeline(image, filename)

    if result and result.success and result.images:
        logging.info(f"Successfully split image with {result.strategy_used}.")
        final_panels = result.images

        # --- THE DEFINITIVE FIX: Revert to specific, correct logic ---
        # Only apply the smart post-processor to the output of ContourAnalysis.
        # No other strategy produces output suitable for it.
        if trim_enabled and result.strategy_used == "contour_analysis":
            logging.info(
                f"Applying subject-aware standardization for '{result.strategy_used}' result..."
            )
            final_panels = standardize_and_center_panels(
                result.images, main_bg_color, padding, config
            )
        else:
            logging.info(
                f"Skipping standardization for '{result.strategy_used}' result as it is not required."
            )

        base_name, ext = os.path.splitext(filename)
        names = ["1_top_left", "2_top_right", "3_bottom_left", "4_bottom_right"]
        for i, panel in enumerate(final_panels):
            output_path = os.path.join(output_dir, f"{base_name}_{names[i]}{ext}")
            save_image(panel, output_path)
    else:
        logging.error(f"Failed to split {filename}: All processing strategies failed.")
