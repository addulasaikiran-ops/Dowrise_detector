# Drowsiness Detector

A real-time Python prototype that uses a webcam to detect prolonged eye closure and trigger an alarm.

## Stack
- Python 3.12
- OpenCV
- MediaPipe Face Landmarker
- NumPy
- Pygame

## Windows setup

This project targets Python 3.12.

1. Open this repository folder in VS Code.
2. Open a PowerShell terminal.
3. Run:

```powershell
.\setup_windows.bat
```

The setup script creates the virtual environment with the installed Astral/uv Python 3.12 runtime.

Then start the detector:

```powershell
.\run.bat
```

On the first run, the program downloads the MediaPipe Face Landmarker model into `models/face_landmarker.task`. An internet connection is required for this first download.

Press **Q** to quit.

## What it does

The prototype tracks one face, calculates an Eye Aspect Ratio (EAR) from eye landmarks, and treats sustained low EAR as prolonged eye closure. When the closure lasts long enough, the status changes to **DROWSY** and an audible alarm starts.

The default thresholds live in `config.py` and can be tuned for your camera and lighting.

> This is a prototype/assistive computer-vision project and is not a safety-certified driver monitoring system.
