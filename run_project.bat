@echo off
setlocal

:: LUNAR FYP - COMPLETE SYSTEM RUNNER
:: Handles dataset, training, and app launch automatically

set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "STREAMLIT=%VENV_DIR%\Scripts\streamlit.exe"

echo ================================================================
echo      LUNAR SURFACE ANALYSIS AI - COMPLETE FYP SYSTEM
echo      SUPARCO Collaboration
echo ================================================================

:: Check environment
if not exist "%VENV_DIR%" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

:: Check data
if not exist "%PROJECT_ROOT%data\pcam" (
    echo [ERROR] Dataset not found!
    echo Please run: .venv\Scripts\python src\data\download_dataset.py
    pause
    exit /b 1
)

:: Generate/update labels
echo [INFO] Checking labeled data...
if not exist "%PROJECT_ROOT%labeled_data\annotations.csv" (
    echo [INFO] Generating labels from dataset...
    "%PYTHON%" src\data\label_importer.py
)

:: Check/train terrain model
echo [INFO] Checking terrain classification model...
if not exist "%PROJECT_ROOT%models\lunar_terrain_classifier.pth" (
    echo [WARN] Terrain model not found. Training recommended.
    choice /C YN /M "Train terrain model now? (Recommended, ~30 min)"
    
    if errorlevel 2 (
        echo [INFO] Skipping training - using untrained weights
    ) else (
        echo [INFO] Starting terrain model training...
        "%PYTHON%" src\models\train_model.py
    )
)

:: Check composition model (optional - uses ImageNet pretrained)
if not exist "%PROJECT_ROOT%models\composition_estimator.pth" (
    echo [INFO] Composition model will use pretrained ImageNet weights
)

:: Check LLM configuration
echo [INFO] Checking LLM configuration...
if exist "%PROJECT_ROOT%.env" (
    findstr /C:"GEMINI_API_KEY" "%PROJECT_ROOT%.env" >nul
    if errorlevel 1 (
        echo [WARN] GEMINI_API_KEY not configured in .env
        echo [INFO] LLM features will be limited. Get a key at: https://ai.google.dev
    ) else (
        echo [OK] LLM configured
    )
) else (
    echo [WARN] .env file not found. Copy .env.example to .env
    echo [INFO] LLM features will be limited.
)

:: Launch application
echo.
echo ================================================================
echo      LAUNCHING SYSTEM
echo ================================================================
echo.
echo The application will open in your browser...
echo Press Ctrl+C to stop the server
echo.

"%PYTHON%" -m streamlit run src\ui\app.py

endlocal
