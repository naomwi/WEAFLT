@echo off
REM Water Quality Forecasting - Installation Script for Windows
REM Run this script to install all required dependencies

echo ============================================
echo Water Quality Forecasting - Installation
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch (CPU version by default)
echo Installing PyTorch...
echo For GPU support, visit: https://pytorch.org/get-started/locally/
python -m pip install torch torchvision torchaudio
echo.

REM Install other requirements
echo Installing other dependencies...
python -m pip install -r requirements.txt
echo.

echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo To run all models:
echo   python run_all.py
echo.
echo To run a specific model:
echo   python run_all.py --model baselines
echo   python run_all.py --model features
echo   python run_all.py --model eventloss
echo   python run_all.py --model full
echo.
echo To list available models:
echo   python run_all.py --list
echo.
pause
