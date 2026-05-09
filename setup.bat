@echo off
REM Quick setup script for RAG Benchmark System (Windows)

echo ================================
echo RAG Benchmark System - Quick Setup
echo ================================
echo.

REM 1. Create .env file if not exists
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo [OK] .env file created. Please add your API keys!
) else (
    echo [OK] .env file already exists
)

REM 2. Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM 3. Install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo Next steps:
echo 1. Edit .env file and add your API keys
echo 2. Activate virtual environment: venv\Scripts\activate.bat
echo 3. Run: python main.py prepare-data
echo.
pause
