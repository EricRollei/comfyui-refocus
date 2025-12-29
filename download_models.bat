@echo off
REM Download Genfocus models for ComfyUI
REM Run this script from the Refocus folder

echo ============================================
echo  Genfocus Model Downloader
echo ============================================
echo.

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)

REM Check HuggingFace login
echo Checking HuggingFace login status...
python -c "from huggingface_hub import HfApi; api = HfApi(); print(f'Logged in as: {api.whoami()[\"name\"]}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Not logged in to HuggingFace
    echo.
    echo To download FLUX.1-dev you need to:
    echo   1. Create account: https://huggingface.co/join
    echo   2. Accept license: https://huggingface.co/black-forest-labs/FLUX.1-dev
    echo   3. Run: huggingface-cli login
    echo.
    set /p LOGIN="Do you want to login now? (y/n): "
    if /i "%LOGIN%"=="y" (
        huggingface-cli login
    )
)

echo.
echo Select what to download:
echo   1. Genfocus LoRAs only (~35MB)
echo   2. FLUX.1-dev only (~23GB)
echo   3. Everything (LoRAs + FLUX)
echo   4. Check status
echo   5. Exit
echo.
set /p CHOICE="Enter choice (1-5): "

if "%CHOICE%"=="1" (
    python scripts\download_models.py --loras-only
) else if "%CHOICE%"=="2" (
    python scripts\download_models.py --flux-only
) else if "%CHOICE%"=="3" (
    python scripts\download_models.py
) else if "%CHOICE%"=="4" (
    python scripts\download_models.py --check
) else if "%CHOICE%"=="5" (
    exit /b 0
) else (
    echo Invalid choice
)

echo.
pause
