from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

# A type alias for clarity throughout the project
Image = np.ndarray


@dataclass
class SplitResult:
    """
    A standardized data structure to hold the results of a splitting attempt.
    This object is returned by every splitting strategy.
    """

    # Was the splitting attempt successful?
    success: bool

    # The name of the strategy that produced this result (e.g., 'midpoint_fallback').
    strategy_used: str

    # A score from 0.0 to 1.0 indicating the strategy's confidence in its result.
    confidence: float

    # A list containing the four resulting image quadrants if successful.
    # Order: [top_left, top_right, bottom_left, bottom_right]
    images: Optional[List[Image]] = None

    # An optional message describing the reason for failure.
    error_message: Optional[str] = None

    # A dictionary to hold any intermediate images or data for debugging purposes.
    # e.g., {'energy_map': energy_map_image, 'contours': contour_image}
    debug_artifacts: Dict[str, Any] = field(default_factory=dict)
