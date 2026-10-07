@echo off
setlocal

echo Creating Python 3.12 virtual environment...
py -3.12 -m venv venv
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
echo Setup failed. Make sure Python 3.12 is installed and available through the Python launcher.
pause
exit /b 1
