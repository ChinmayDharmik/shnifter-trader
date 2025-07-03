@echo off
REM =============================================
REM Start The Shnifter Trader using the virtual environment (Updated July 2025)
REM =============================================

REM Check if the virtual environment exists
if exist .venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo [INFO] Running The Shnifter Trader main script...
    .venv\Scripts\python Multi_Model_Trading_Bot.py
    echo [INFO] The Shnifter Trader has exited.
) else (
    echo [ERROR] .venv not found. Please run start_shnifter_setup.bat or setup_python313.bat first.
)

REM Pause to keep the window open for user review
pause