#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Install SAM 2 from GitHub
echo "Installing SAM 2..."
pip install git+https://github.com/facebookresearch/sam2.git

echo "Setup complete! Activate the environment with 'source .venv/bin/activate'"
