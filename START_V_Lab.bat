@echo off
REM ============================================================
REM Enhanced ROI Distance Visualizer V_Lab Launcher
REM This script launches the V_Lab version with enhanced mask visualization
REM ============================================================

echo.
echo ============================================================
echo Enhanced ROI Distance Visualizer V_Lab
echo Enhanced Mask Visualization with Centroid Overlay
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

REM Check if pip is available
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please install pip or reinstall Python with pip included
    pause
    exit /b 1
)

echo Checking dependencies...
echo.

REM Check each package individually
python -c "import pandas" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] pandas
) else (
    echo [MISSING] pandas
    set MISSING=1
)

python -c "import numpy" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] numpy
) else (
    echo [MISSING] numpy
    set MISSING=1
)

python -c "import PyQt5" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] PyQt5
) else (
    echo [MISSING] PyQt5
    set MISSING=1
)

python -c "import matplotlib" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] matplotlib
) else (
    echo [MISSING] matplotlib
    set MISSING=1
)

python -c "import scipy" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] scipy
) else (
    echo [MISSING] scipy
    set MISSING=1
)

python -c "import skimage" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] scikit-image
) else (
    echo [MISSING] scikit-image
    set MISSING=1
)

python -c "import pydicom" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] pydicom
) else (
    echo [MISSING] pydicom
    set MISSING=1
)

python -c "import cupy" >nul 2>&1
if errorlevel 1 (
    echo [OPTIONAL] cupy (GPU support not available - will use CPU)
) else (
    echo [OK] cupy (GPU support available^)
)

python -c "import openpyxl" >nul 2>&1
if errorlevel 1 (
    echo [OPTIONAL] openpyxl (Excel export not available - install with: pip install openpyxl^)
) else (
    echo [OK] openpyxl (Excel export available^)
)

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [OPTIONAL] torch (CUDA DLL path detection not available^)
) else (
    echo [OK] torch (CUDA DLL path detection available^)
)

echo.

REM Install missing packages if any
if defined MISSING (
    echo Some packages are missing. Installing dependencies...
    echo This may take a few minutes...
    python -m pip install --upgrade pip --quiet
    python -m pip install pandas numpy PyQt5 matplotlib scipy scikit-image pydicom --quiet
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo Dependencies installed successfully!
    echo.
)

REM Launch the application
python roi_distance_visualizer_enhanced_V_Lab.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Application failed to start
    echo Check the error messages above for details
    pause
    exit /b 1
)

