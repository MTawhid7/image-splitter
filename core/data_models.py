from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

Image = np.ndarray


@dataclass
class SplitResult:
    """
    A standardized data structure to hold the results of a splitting attempt.
    """

    success: bool
    strategy_used: str
    confidence: float
    images: Optional[List[Image]] = None
    # --- THE UPGRADE: Add a definitive list of content bounds ---
    # Each tuple is (x, y, w, h) relative to the full original image.
    bounds: Optional[List[Tuple[int, int, int, int]]] = None
    error_message: Optional[str] = None
    debug_artifacts: Dict[str, Any] = field(default_factory=dict)
