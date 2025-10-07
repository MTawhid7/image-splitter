# --- Configuration ---
PYTHON = python
INPUT_DIR = input_images
OUTPUT_DIR = output_results

# --- Commands ---
.PHONY: install run clean clean-output

install:
	@echo "Creating virtual environment and installing dependencies..."
	$(PYTHON) -m venv venv
	@. venv/bin/activate; pip install -r requirements.txt
	@echo "Installation complete. Activate the venv with: source venv/bin/activate"

run:
	@echo "Running the image splitter..."
	$(PYTHON) main.py --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)

clean-output:
	@echo "Clearing output directory: $(OUTPUT_DIR)"
	rm -f $(OUTPUT_DIR)/*
	# --- THE FIX: Also remove the debug directory ---
	rm -rf $(OUTPUT_DIR)/debug

clean:
	@echo "Cleaning up the project (venv, pycache, output)..."
	rm -rf venv
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -f $(OUTPUT_DIR)/*
	# --- THE FIX: Also remove the debug directory ---
	rm -rf $(OUTPUT_DIR)/debug
	@echo "Cleanup complete."

test:
	@echo "Test suite not yet implemented."