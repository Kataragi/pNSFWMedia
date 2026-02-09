@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo  pNSFWMedia - Windows Setup Script
echo ============================================================
echo.

:: Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.10 or 3.11 from https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [INFO] Python version: %PYVER%

:: Create virtual environment
set VENV_DIR=venv
if exist %VENV_DIR% (
    echo [INFO] Virtual environment already exists: %VENV_DIR%
    set /p RECREATE="Recreate virtual environment? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo [INFO] Removing existing virtual environment...
        rmdir /s /q %VENV_DIR%
        echo [INFO] Creating new virtual environment...
        python -m venv %VENV_DIR%
    )
) else (
    echo [INFO] Creating virtual environment: %VENV_DIR%
    python -m venv %VENV_DIR%
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

:: Upgrade pip
echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo.
echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install some packages.
    echo         Please check the error messages above.
    pause
    exit /b 1
)

:: Install CLIP (required for adversarial perturbation)
echo.
echo [INFO] Installing OpenAI CLIP...
pip install setuptools
pip install git+https://github.com/openai/CLIP.git

if errorlevel 1 (
    echo [WARNING] Failed to install CLIP.
    echo           Adversarial perturbation features may not work.
    echo           You can try manually: pip install git+https://github.com/openai/CLIP.git
)

:: Create necessary directories
echo.
echo [INFO] Creating directory structure...
if not exist models mkdir models
if not exist models\adversarial mkdir models\adversarial
if not exist dataset mkdir dataset
if not exist dataset\images mkdir dataset\images
if not exist dataset\images\sfw mkdir dataset\images\sfw
if not exist dataset\images\nsfw mkdir dataset\images\nsfw
if not exist dataset\embeddings mkdir dataset\embeddings
if not exist output mkdir output
if not exist logs mkdir logs
if not exist results mkdir results

:: Summary
echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo  Virtual environment: %VENV_DIR%
echo.
echo  To activate the environment:
echo    %VENV_DIR%\Scripts\activate
echo.
echo  Quick start (adversarial perturbation inference):
echo    python src/adversarial/apply.py ^
echo        --checkpoint models/adversarial/sfm_final.pt ^
echo        --image path/to/image.jpg
echo.
echo  For CUDA/GPU support, install PyTorch manually:
echo    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.
echo ============================================================

pause
