#!/bin/bash
# setup_unix.sh

echo "Creating Virtual Environment..."
python3 -m venv .venv

echo "Installing Requirements..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo "Setup Complete! To activate your environment, run: source .venv/bin/activate"