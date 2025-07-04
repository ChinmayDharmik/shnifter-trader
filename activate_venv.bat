@echo off
REM =============================================
REM Activate the .venv virtual environment for The Shnifter Trader (Updated July 2025)
REM =============================================

REM Try to detect the shell and activate accordingly
if exist .venv\Scripts\activate.bat (
    echo [INFO] Detected Command Prompt (cmd.exe) or compatible shell.
    call .venv\Scripts\activate.bat
    echo [INFO] .venv is now active. You can now run Python commands in this shell.
    echo [INFO] To launch the Shnifter Trader frontend, run:
    echo     py Multi_Model_Trading_Bot.py
    cmd /k
) else (
    if exist .venv\Scripts\Activate.ps1 (
        echo [INFO] Detected PowerShell. Please run the following command manually:
        echo     .venv\Scripts\Activate.ps1
        echo If you see a policy error, run PowerShell as Administrator and execute:
        echo     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
        echo [INFO] .venv is now active. You can now run Python commands in this shell.
        echo [INFO] To launch the Shnifter Trader frontend, run:
        echo     py Multi_Model_Trading_Bot.py
        pause
        exit /b 1
    ) else (
        echo [ERROR] No activation script found. Please run start_shnifter_setup.bat or setup_python313.bat first.
        pause
        exit /b 1
    )
)
