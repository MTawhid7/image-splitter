import logging
from typing import List
import numpy as np

from core.data_models import Image
from utils.image_utils import find_content_bounds


def get_dominant_background_color(panel: Image) -> List[int]:
    """
    Determines the background color by finding the median of the four corner pixels.
    This is extremely robust for both padded and tightly-cropped images.
    """
    try:
        h, w, _ = panel.shape
        # --- THE FINAL, CRITICAL FIX: Sample all four corners ---
        corners = [
            panel[0, 0],  # Top-left
            panel[0, w - 1],  # Top-right
            panel[h - 1, 0],  # Bottom-left
            panel[h - 1, w - 1],  # Bottom-right
        ]
        # The median color will ignore any single outlier pixel that is part of the content.
        median_color = np.median(corners, axis=0)
        return median_color.astype(int).tolist()
    except Exception as e:
        logging.warning(
            f"Could not determine dominant background color due to: {e}. Defaulting to white."
        )
        return [255, 255, 255]


def standardize_panels(panels: List[Image], padding: int, config: dict) -> List[Image]:
    """
    The definitive post-processing function. It standardizes all panels to the
    exact same final dimensions with content perfectly centered on a new canvas
    that matches the original background color of each panel.
    """
    all_bounds = [find_content_bounds(p, config) for p in panels]
    valid_bounds = [b for b in all_bounds if b is not None and b[2] > 0 and b[3] > 0]

    if not valid_bounds:
        logging.warning(
            "Could not find content in any panel for standardization. Returning original panels."
        )
        return panels

    max_w = max(b[2] for b in valid_bounds)
    max_h = max(b[3] for b in valid_bounds)
    canvas_w = max_w + (padding * 2)
    canvas_h = max_h + (padding * 2)
    logging.info(
        f"Standardizing all panels to a final canvas size of: {canvas_w}x{canvas_h}"
    )

    final_panels = []
    for i, panel in enumerate(panels):
        bounds = all_bounds[i]

        background_color = get_dominant_background_color(panel)
        canvas = np.full((canvas_h, canvas_w, 3), background_color, dtype=np.uint8)

        if bounds is None:
            final_panels.append(canvas)
            continue

        x, y, w, h = bounds
        content = panel[y : y + h, x : x + w]

        paste_x = (canvas_w - w) // 2
        paste_y = (canvas_h - h) // 2

        canvas[paste_y : paste_y + h, paste_x : paste_x + w] = content
        final_panels.append(canvas)

    return final_panels

