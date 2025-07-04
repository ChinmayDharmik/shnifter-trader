@echo off
REM =============================================
REM Shnifter Python 3.13 Environment Setup Script (Updated July 2025)
REM =============================================

REM Use py launcher for Python 3.13
py -3.13 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.13 not found. Please install Python 3.13 from https://www.python.org/downloads/.
    pause
    exit /b 1
)

REM Remove old venv if exists
if exist .venv rmdir /s /q .venv

REM Create new venv with Python 3.13
py -3.13 -m venv .venv
if exist .venv (
    echo [SUCCESS] Created .venv with Python 3.13.
) else (
    echo [ERROR] Failed to create .venv.
    pause
    exit /b 1
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Upgrade pip, setuptools, wheel
python -m pip install --upgrade pip setuptools wheel

REM Install requirements
python -m pip install -r requirements.txt

echo [INFO] Python 3.13 environment setup complete. You can now run start_shnifter.bat.
pause
