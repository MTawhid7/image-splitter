import logging
import importlib
from core.data_models import SplitResult, Image


class ImageSplitter:
    """
    Orchestrates the image splitting process by managing a pipeline of strategies.
    """

    def __init__(self, config: dict):
        self.config = config
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> list:
        """
        Dynamically loads strategy classes based on the pipeline specified in config.yaml.
        This makes the system plug-and-play.
        """
        loaded_strategies = []
        pipeline_names = self.config.get("strategy_pipeline", [])
        debug_mode = self.config.get("debug_mode", False)

        logging.info(f"Loading strategy pipeline: {pipeline_names}")

        for name in pipeline_names:
            try:
                # e.g., 'midpoint_fallback' -> 'MidpointFallbackStrategy'
                class_name = (
                    "".join(word.capitalize() for word in name.split("_")) + "Strategy"
                )
                # Module path e.g., 'strategies.midpoint_fallback'
                module = importlib.import_module(f"strategies.{name}")
                StrategyClass = getattr(module, class_name)

                # Get the specific config for this strategy
                strategy_config = self.config.get(name, {})

                # Instantiate the strategy and add it to our list
                loaded_strategies.append(StrategyClass(strategy_config, debug_mode))
                logging.debug(f"Successfully loaded strategy: {class_name}")

            except (ImportError, AttributeError) as e:
                logging.error(
                    f"Could not load strategy '{name}': {e}. Please check the class/file names."
                )

        return loaded_strategies

    def split(self, image: Image) -> SplitResult:
        """
        Attempts to split an image by trying each loaded strategy in order.

        Args:
            image (Image): The image to split.

        Returns:
            SplitResult: The first successful result from a strategy that meets its
                         confidence threshold.
        """
        if not self.strategies:
            logging.error("No strategies were loaded. Cannot perform split.")
            return SplitResult(
                success=False,
                strategy_used="none",
                confidence=0.0,
                error_message="No strategies loaded.",
            )

        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            logging.info(f"Attempting to split with strategy: {strategy_name}")

            result = strategy.split(image)

            # Get the confidence threshold for this specific strategy from the config
            threshold = self.config.get(result.strategy_used, {}).get(
                "confidence_threshold", 0.8
            )

            if result.success and result.confidence >= threshold:
                logging.info(
                    f"Success! Strategy '{strategy_name}' succeeded with confidence {result.confidence:.2f} (Threshold: {threshold})."
                )
                return result
            else:
                logging.warning(
                    f"Strategy '{strategy_name}' did not meet criteria (Success: {result.success}, Confidence: {result.confidence:.2f}, Threshold: {threshold}). Trying next strategy."
                )

        logging.error(
            "All strategies in the pipeline failed to produce a confident result."
        )
        return SplitResult(
            success=False,
            strategy_used="pipeline_exhausted",
            confidence=0.0,
            error_message="All strategies failed.",
        )
