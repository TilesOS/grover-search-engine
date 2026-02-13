#!/bin/bash

# Grover Search Engine - Setup Script
# This script sets up the environment and installs all dependencies

set -e  # Exit on error

echo "========================================="
echo "Grover Search Engine - Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing..."
    rm -rf venv
fi

python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the demo:"
echo "  python grover_search.py"
echo ""
echo "To run tutorials:"
echo "  python examples.py"
echo ""
echo "To run tests:"
echo "  pytest test_grover_search.py -v"
echo ""
echo "Happy Quantum Computing!"
