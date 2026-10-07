@echo off
setlocal

echo Finding Python 3.12...

for /f "delims=" %%P in ('py -V:Astral/CPython3.12.13 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON312=%%P"

if not defined PYTHON312 (
    echo Could not find the Astral/uv Python 3.12 runtime.
    echo Please run:
    echo   py --list
    echo and verify Python 3.12 is installed.
    goto :error
)

echo Using Python:
echo %PYTHON312%

echo Creating Python 3.12 virtual environment...
"%PYTHON312%" -m venv venv
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
echo Setup failed.
pause
exit /b 1
