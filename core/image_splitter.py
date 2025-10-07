import logging
import importlib
from core.data_models import SplitResult, Image


class ImageSplitter:
    def __init__(self, config: dict):
        self.config = config
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> dict:
        loaded_strategies = {}
        pipeline_names = self.config.get("strategy_pipeline", [])
        logging.info(f"Attempting to load strategies: {pipeline_names}")
        for name in pipeline_names:
            try:
                class_name = (
                    "".join(word.capitalize() for word in name.split("_")) + "Strategy"
                )
                module = importlib.import_module(f"strategies.{name}")
                StrategyClass = getattr(module, class_name)
                strategy_config = self.config.get(name, {})
                if "projection_profile" in name:
                    strategy_config.update(self.config.get("projection_profile", {}))
                loaded_strategies[name] = StrategyClass(
                    strategy_config, self.config.get("debug_mode", False)
                )
                logging.debug(f"Successfully loaded strategy: {class_name}")
            except (ImportError, AttributeError) as e:
                logging.error(f"Could not load strategy '{name}': {e}.")
        return loaded_strategies

    def get_strategy(self, name: str):
        strategy = self.strategies.get(name)
        if not strategy:
            logging.error(f"Attempted to get non-existent strategy: '{name}'")
        return strategy

    def run_full_pipeline(self, image: Image, filename: str) -> SplitResult:
        """Runs the linear failover pipeline, passing filename for debugging."""
        fallback_order = ["contour_analysis", "midpoint_fallback"]
        for name in fallback_order:
            strategy = self.get_strategy(name)
            if not strategy:
                continue

            logging.info(
                f"Fallback Pipeline: Attempting strategy: {strategy.__class__.__name__}"
            )
            # --- FIX: Pass 'filename' to the split method ---
            result = strategy.split(image, filename)

            threshold = self.config.get(name, {}).get("confidence_threshold", 0.8)
            if result.success and result.confidence >= threshold:
                return result

        logging.error("All strategies in the fallback pipeline failed.")
        return SplitResult(
            success=False,
            strategy_used="pipeline_exhausted",
            confidence=0.0,
            error_message="All strategies failed.",
        )
