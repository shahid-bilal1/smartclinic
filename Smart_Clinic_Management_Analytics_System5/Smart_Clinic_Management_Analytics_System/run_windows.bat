@echo off
REM One-click start for Windows. Double-click this file.
cd /d "%~dp0"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing required packages (first time only)...
pip install -r requirements.txt
echo.
echo Starting Smart Clinic... open http://127.0.0.1:5000 in your browser.
python app.py
pause
