#!/bin/bash
# Water Quality Forecasting - Installation Script for Linux/macOS
# Run: chmod +x install.sh && ./install.sh

echo "============================================"
echo "Water Quality Forecasting - Installation"
echo "============================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.8+ first"
    exit 1
fi

echo "Python found:"
python3 --version
echo

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip
echo

# Install PyTorch (CPU version by default)
echo "Installing PyTorch..."
echo "For GPU support, visit: https://pytorch.org/get-started/locally/"
python3 -m pip install torch torchvision torchaudio
echo

# Install other requirements
echo "Installing other dependencies..."
python3 -m pip install -r requirements.txt
echo

echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo
echo "To run all models:"
echo "  python3 run_all.py"
echo
echo "To run a specific model:"
echo "  python3 run_all.py --model baselines"
echo "  python3 run_all.py --model features"
echo "  python3 run_all.py --model eventloss"
echo "  python3 run_all.py --model full"
echo
echo "To list available models:"
echo "  python3 run_all.py --list"
echo
