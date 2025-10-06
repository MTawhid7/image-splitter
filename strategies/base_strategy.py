from abc import ABC, abstractmethod
from core.data_models import SplitResult, Image


class BaseSplittingStrategy(ABC):
    """
    Abstract Base Class (Interface) for all splitting strategies.

    This class defines the contract that every splitting algorithm must follow.
    It ensures they are interchangeable and can be used in the pipeline.
    """

    def __init__(self, config: dict, debug: bool = False):
        """
        Initializes the strategy with its specific configuration.

        Args:
            config (dict): The configuration dictionary for this specific strategy,
                         loaded from config.yaml.
            debug (bool): Flag indicating if debug mode is active.
        """
        self.config = config
        self.debug = debug

    @abstractmethod
    def split(self, image: Image) -> SplitResult:
        """
        The core method that attempts to split the input image.
        This method MUST be implemented by all concrete strategy classes.

        Args:
            image (Image): The input image (as a NumPy array) to be split.

        Returns:
            SplitResult: An object containing the outcome of the splitting attempt.
        """
        pass
