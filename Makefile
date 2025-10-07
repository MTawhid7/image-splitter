# --- Configuration ---
PYTHON = python
INPUT_DIR = input_images
OUTPUT_DIR = output_results
DEBUG_DIR = $(OUTPUT_DIR)/debug

# --- Commands ---
.PHONY: install run clean

install:
	@echo "Creating virtual environment and installing dependencies..."
	$(PYTHON) -m venv venv
	@. venv/bin/activate; pip install -r requirements.txt
	@echo "Installation complete. Activate the venv with: source venv/bin/activate"

run:
	@echo "Running the image splitter..."
	$(PYTHON) main.py --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)

clean:
	@echo "Clearing output directory: $(OUTPUT_DIR)"
	# Remove everything (files and directories) inside output_results safely
	rm -rf $(OUTPUT_DIR)/*
	# Optional: also ensure the debug directory is cleared (if recreated elsewhere)
	rm -rf $(DEBUG_DIR)

test:
	@echo "Test suite not yet implemented."