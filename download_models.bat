@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo  pNSFWMedia - Download Pre-trained Models from HuggingFace
echo ============================================================
echo.

:: Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo [WARNING] Virtual environment is not activated.
    echo           Activating venv...
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    ) else (
        echo [ERROR] Virtual environment not found.
        echo         Please run setup.bat first.
        pause
        exit /b 1
    )
)

:: Install huggingface_hub if not installed
echo [INFO] Checking huggingface_hub...
pip show huggingface_hub >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing huggingface_hub...
    pip install huggingface_hub
)

:: Create models directory
if not exist models mkdir models
if not exist models\adversarial mkdir models\adversarial

echo.
echo [INFO] Downloading models from HuggingFace...
echo        Repository: kataragi/adversarial
echo.

:: Download using huggingface-cli
huggingface-cli download kataragi/adversarial --local-dir models/adversarial --local-dir-use-symlinks False

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to download models.
    echo         Please check your internet connection.
    echo.
    echo         Alternative manual download:
    echo         https://huggingface.co/kataragi/adversarial/tree/main
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Download Complete!
echo ============================================================
echo.
echo  Models saved to: models\adversarial\
echo.

:: List downloaded files
echo  Downloaded files:
dir /b models\adversarial\*.pt 2>nul
if errorlevel 1 (
    echo    (no .pt files found)
)

echo.
echo  Quick start:
echo    python src/adversarial/apply.py ^
echo        --checkpoint models/adversarial/sfm_final.pt ^
echo        --image path/to/image.jpg
echo.
echo ============================================================

pause
