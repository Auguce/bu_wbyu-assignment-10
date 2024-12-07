# Makefile for Assignment 10: Image Search

.PHONY: run clean setup

# Default target to run the Flask app
run:
	@echo "Starting the Flask app..."
	@python app.py

# Setup target to create a virtual environment and install dependencies
setup:
	@echo "Setting up the environment..."
	@python -m venv venv
	@venv/Scripts/pip install --upgrade pip
	@venv/Scripts/pip install -r requirements.txt

# Clean target to remove unnecessary files
clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ uploaded/ venv/ .pytest_cache
