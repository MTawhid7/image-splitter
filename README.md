# Algorithmic Image Splitter

This project is a sophisticated Python application designed to programmatically split composite images into four distinct panels. It uses a cascading pipeline of computer vision algorithms to robustly handle a variety of image layouts, including those with clear dividers, no dividers, and inconsistent backgrounds.

## Features

- **Modular Strategy Pipeline:** Easily add, remove, or reorder splitting algorithms.
- **Robust Algorithmic Approach:** Combines multiple computer vision techniques for high accuracy without requiring machine learning.
- **Configuration Driven:** All algorithmic parameters and settings are managed in a `config.yaml` file for easy tuning.
- **Automated Validation:** Each strategy self-validates its results, ensuring only high-confidence splits are accepted.
- **Graceful Fallbacks:** If advanced strategies fail, the system defaults to simpler methods to prevent crashes.

## Current Strategies Implemented

1.  **Projection Profile:** Detects the precise bounds of prominent horizontal and vertical divider lines using robust band-scanning. Ideal for images with clear separators.
2.  **Contour Analysis:** Identifies the four main content areas in images without dividers by grouping all content contours into quadrants.
3.  **Midpoint Fallback:** A simple, reliable strategy that splits the image into four equal quadrants as a last resort.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MTawhid7/image-splitter.git
    cd image-splitter
    ```

2.  **Create the virtual environment and install dependencies:**
    This project uses a `Makefile` for convenience.
    ```bash
    make install
    ```
    This will create a `venv` folder and install the packages listed in `requirements.txt`.

3.  **Activate the virtual environment:**
    -   **macOS/Linux:** `source venv/bin/activate`
    -   **Windows:** `.\venv\Scripts\activate`

## How to Run

1.  **Place your composite images** into the `input_images` directory.
2.  **Run the application** using the Makefile command:
    ```bash
    make run
    ```
3.  The split images will be saved in the `output_results` directory.

You can also specify different input/output directories:
```bash
make run INPUT_DIR=path/to/my_input OUTPUT_DIR=path/to/my_output
```

## How to Clean
To clear only the output images:
```bash
make clean-output
```
To remove the virtual environment, pycache, and output:
```bash
make clean
```