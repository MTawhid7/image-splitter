# Makefile for the Image Splitter Project

# --- Configuration ---
PYTHON = python
INPUT_DIR = input_images
OUTPUT_DIR = output_results

# --- Commands ---

# Phony targets are not actual files. This prevents conflicts.
.PHONY: install run clean test clean-output

install:
	@echo "Creating virtual environment and installing dependencies..."
	$(PYTHON) -m venv venv
	@. venv/bin/activate; pip install -r requirements.txt
	@echo "Installation complete. Activate the venv with: source venv/bin/activate"

run:
	@echo "Running the image splitter..."
	$(PYTHON) main.py --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)

# NEW COMMAND to quickly clear the output folder
clean-output:
	@echo "Clearing output directory: $(OUTPUT_DIR)"
	rm -f $(OUTPUT_DIR)/*

clean:
	@echo "Cleaning up the project (venv, pycache, output)..."
	rm -rf venv
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -f $(OUTPUT_DIR)/*
	@echo "Cleanup complete."

test:
	@echo "Test suite not yet implemented."