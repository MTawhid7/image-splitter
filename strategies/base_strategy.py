from abc import ABC, abstractmethod
from core.data_models import SplitResult, Image


class BaseSplittingStrategy(ABC):
    """
    Abstract Base Class (Interface) for all splitting strategies.
    """

    def __init__(self, config: dict, debug: bool = False):
        self.config = config
        self.debug = debug

    @abstractmethod
    def split(self, image: Image, filename: str) -> SplitResult:
        """
        The core method that attempts to split the input image.
        This method MUST be implemented by all concrete strategy classes.

        Args:
            image (Image): The input image (as a NumPy array) to be split.
            filename (str): The original filename, used for unique debug outputs.

        Returns:
            SplitResult: An object containing the outcome of the splitting attempt.
        """
        pass
