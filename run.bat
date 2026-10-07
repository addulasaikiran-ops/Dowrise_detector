@echo off
call venv\Scripts\activate
if errorlevel 1 (
  echo Virtual environment not found.
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
python ui.py
pause
