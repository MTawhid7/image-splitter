import logging
import sys


def setup_logging(debug: bool = False):
    """
    Configures the root logger for the application.

    Args:
        debug (bool): If True, sets the logging level to DEBUG for more verbose output.
                      Otherwise, sets it to INFO.
    """
    log_level = logging.DEBUG if debug else logging.INFO

    # Get the root logger
    logger = logging.getLogger()

    # Clear existing handlers to avoid duplicate logs if this is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler to print to the console (standard output)
    handler = logging.StreamHandler(sys.stdout)

    # Create a formatter and set it for the handler
    # Format: [TIMESTAMP] [LOG_LEVEL] [MODULE_NAME] MESSAGE
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Set the logger's level
    logger.setLevel(log_level)

    logging.info("Logging configured successfully.")
    if debug:
        logging.debug("Debug mode is enabled.")
