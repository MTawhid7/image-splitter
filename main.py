import os
import argparse
import yaml
import logging
import sys

from utils.logging_config import setup_logging
from utils.image_utils import load_image
from core.image_splitter import ImageSplitter
from core.processing import process_image  # <-- IMPORT THE NEW ENGINE


def load_config(config_path: str) -> dict | None:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load or parse config file at {config_path}: {e}")
        return None


def main(args):
    """
    The main entry point. Handles setup and file iteration, but delegates
    the core logic to the processing module.
    """
    config = load_config(args.config)
    if config is None:
        sys.exit(1)

    setup_logging(debug=config.get("debug_mode", False))

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Instantiate the splitter once
    splitter = ImageSplitter(config)

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(args.input_dir, filename)
            logging.info(f"--- Processing image: {filename} ---")

            image = load_image(image_path)
            if image is not None:
                # Delegate all the hard work to our new processing function
                process_image(image, filename, splitter, config, args.output_dir)

            print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split composite images into four quadrants."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory for input images."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for output images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    main(args)
