#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies
apt-get update
apt-get install -y libzbar0 libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Print versions for debugging
python --version
pip --version
echo "System dependencies installed successfully"
