@echo off
REM ===============================
REM The Shnifter Trader Setup Script (Updated July 2025)
REM ===============================

REM Set up log file for all actions
set LOGFILE=shnifter_setup_log.txt

REM Start logging
echo [INFO] Starting The Shnifter Trader setup... > %LOGFILE%
echo [INFO] Creating Python 3.13 virtual environment... >> %LOGFILE%

REM Create virtual environment using Python 3.13
py -3.13 -m venv .venv 2>> %LOGFILE%
if exist .venv (
    echo [SUCCESS] Virtual environment created. >> %LOGFILE%
    echo [SUCCESS] .venv directory found after creation. >> %LOGFILE%
) else (
    echo [ERROR] Failed to create virtual environment. >> %LOGFILE%
    echo [ERROR] .venv directory not found. Exiting setup. >> %LOGFILE%
    exit /b 1
)

REM Activate the virtual environment
echo [INFO] Activating virtual environment... >> %LOGFILE%
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment. >> %LOGFILE%
    exit /b 1
)

REM Upgrade pip, setuptools, wheel
echo [INFO] Upgrading pip, setuptools, and wheel... >> %LOGFILE%
python -m pip install --upgrade pip setuptools wheel >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to upgrade pip/setuptools/wheel. >> %LOGFILE%
    exit /b 1
) else (
    echo [SUCCESS] pip, setuptools, and wheel upgraded. >> %LOGFILE%
)

REM Install requirements
echo [INFO] Installing Python dependencies from requirements.txt... >> %LOGFILE%
python -m pip install -r requirements.txt >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements. >> %LOGFILE%
    exit /b 1
) else (
    echo [SUCCESS] Requirements installed. >> %LOGFILE%
)

echo [INFO] Setup complete. You can now run start_shnifter.bat to launch the app. >> %LOGFILE%
pause
