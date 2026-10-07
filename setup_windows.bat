@echo off
setlocal

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 goto :error

echo Installing dependencies...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete.
echo Run the project with: run.bat
pause
exit /b 0

:error
echo.
echo Setup failed. Make sure Python 3.11 is installed and available in PATH.
pause
exit /b 1
