@echo off
REM =============================================
REM Open a new Command Prompt with the .venv environment activated
REM =============================================

start cmd /k ".venv\Scripts\activate.bat && echo [INFO] .venv is now active. To launch the Shnifter Trader frontend, run: && echo     py Multi_Model_Trading_Bot.py && echo [INFO] Or, to run the Multi-Modal Trading Bot, use: && echo     py Multi_Model_Trading_Bot.py"
