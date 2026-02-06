@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo  pNSFWMedia - Gradio GUI
echo ============================================================
echo.

:: Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo [INFO] Activating virtual environment...
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    ) else (
        echo [ERROR] Virtual environment not found.
        echo         Please run setup.bat first.
        pause
        exit /b 1
    )
)

:: Check if models exist
if not exist models\adversarial\high_noise.pt (
    echo [WARNING] Models not found in models\adversarial\
    echo           Please run download_models.bat first.
    echo.
)

echo [INFO] Starting Gradio GUI...
echo        Open http://localhost:7860 in your browser
echo.
echo        Press Ctrl+C to stop the server
echo ============================================================
echo.

python src/adversarial/gui.py

pause
