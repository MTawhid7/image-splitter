import os
import argparse
import yaml
import logging
import sys
from typing import Optional
from utils.logging_config import setup_logging
from utils.image_utils import load_image, save_image
from core.image_splitter import ImageSplitter  # <--- IMPORT THE ORCHESTRATOR


def load_config(config_path: str) -> Optional[dict]:
    """Loads the YAML configuration file."""
    # (No changes to this function)
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return None


def main(args):
    """Main function to run the image splitter application."""
    config = load_config(args.config)
    if config is None:
        sys.exit(1)

    setup_logging(debug=config.get("debug_mode", False))

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not os.path.isdir(args.output_dir):
        logging.info(f"Output directory not found. Creating it now: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    # --- CORE LOGIC UPDATE ---
    # 1. Instantiate the ImageSplitter
    splitter = ImageSplitter(config)

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(args.input_dir, filename)
            logging.info(f"--- Processing image: {filename} ---")

            image = load_image(image_path)

            if image is not None:
                # 2. Call the main split method
                result = splitter.split(image)

                # 3. Handle the result
                if result.success and result.images:
                    logging.info(
                        f"Successfully split image with {result.strategy_used}."
                    )
                    base_name, ext = os.path.splitext(filename)

                    # Define output names
                    names = [
                        "1_top_left",
                        "2_top_right",
                        "3_bottom_left",
                        "4_bottom_right",
                    ]

                    # Save each of the four images
                    for i, split_img in enumerate(result.images):
                        output_path = os.path.join(
                            args.output_dir, f"{base_name}_{names[i]}{ext}"
                        )
                        save_image(split_img, output_path)
                else:
                    logging.error(
                        f"Failed to split {filename}. Reason: {result.error_message}"
                    )

            print("-" * 50)


if __name__ == "__main__":
    # (No changes to the argparse section)
    parser = argparse.ArgumentParser(
        description="Split composite images into four quadrants."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing images to be split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where split images will be saved.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    main(args)
