import logging
import os
from typing import List, Tuple
import numpy as np

from classifier.diagnostics import ImageType
from classifier.image_classifier import ImageClassifier
from core.data_models import Image, SplitResult
from core.image_splitter import ImageSplitter
from utils.image_utils import save_image


def standardize_and_center_panels(
    panels: List[Image], bounds: List[Tuple[int, int, int, int]], padding: int
) -> List[Image]:
    """
    Standardizes panels to a consistent size with content perfectly centered.
    This should ONLY be used for strategies like ContourAnalysis.
    """
    valid_bounds = [b for b in bounds if b is not None and b[2] > 0 and b[3] > 0]
    if not valid_bounds:
        return panels

    max_w = max(b[2] for b in valid_bounds)
    max_h = max(b[3] for b in valid_bounds)
    canvas_w, canvas_h = max_w + (padding * 2), max_h + (padding * 2)
    logging.info(f"Standardizing all panels to a canvas of {canvas_w}x{canvas_h}")

    final_panels = []
    for i, panel in enumerate(panels):
        h_panel, w_panel, _ = panel.shape
        corners = [
            panel[0, 0],
            panel[0, w_panel - 1],
            panel[h_panel - 1, 0],
            panel[h_panel - 1, w_panel - 1],
        ]
        background_color = np.median(corners, axis=0).astype(int).tolist()
        canvas = np.full((canvas_h, canvas_w, 3), background_color, dtype=np.uint8)

        panel_bounds = bounds[i]
        if panel_bounds and panel_bounds[2] > 0:
            x, y, w, h = panel_bounds
            content = panel[y : y + h, x : x + w]
            paste_x, paste_y = (canvas_w - w) // 2, (canvas_h - h) // 2
            canvas[paste_y : paste_y + h, paste_x : paste_x + w] = content
        final_panels.append(canvas)
    return final_panels


def process_image(
    image: Image, filename: str, splitter: ImageSplitter, config: dict, output_dir: str
):
    trim_config = config.get("trimming", {})
    trim_enabled = trim_config.get("enabled", False)
    padding = trim_config.get("padding", 15)

    classifier = ImageClassifier(config)
    image_type = classifier.diagnose(image)

    strategy_name = None
    if image_type == ImageType.DIVIDERS_FULL:
        strategy_name = "projection_profile"
    elif image_type in [ImageType.SEAMLESS_UNIFORM, ImageType.SEAMLESS_COMPLEX]:
        strategy_name = "contour_analysis"

    result: SplitResult | None = None
    if strategy_name:
        logging.info(
            f"Executing primary strategy for {image_type.name}: {strategy_name}"
        )
        strategy = splitter.get_strategy(strategy_name)
        if strategy:
            result = strategy.split(image, filename)

    if not (result and result.success):
        logging.warning("Primary strategy failed. Running fallback pipeline.")
        result = splitter.run_full_pipeline(image, filename)

    if result and result.success and result.images:
        logging.info(f"Successfully split image with {result.strategy_used}.")
        final_panels = result.images

        # --- THE DEFINITIVE FIX: CONDITIONAL POST-PROCESSING ---
        # Only standardize if the strategy was contour-based.
        # Projection-based strategies are already perfectly aligned.
        if (
            trim_enabled
            and result.strategy_used == "contour_analysis"
            and result.bounds
        ):
            logging.info(
                "Applying standardization and centering for contour-based result..."
            )
            final_panels = standardize_and_center_panels(
                result.images, result.bounds, padding
            )
        else:
            logging.info("Skipping standardization for projection-based result.")

        base_name, ext = os.path.splitext(filename)
        names = ["1_top_left", "2_top_right", "3_bottom_left", "4_bottom_right"]
        for i, panel in enumerate(final_panels):
            output_path = os.path.join(output_dir, f"{base_name}_{names[i]}{ext}")
            save_image(panel, output_path)
    else:
        logging.error(f"Failed to split {filename}: All processing strategies failed.")
