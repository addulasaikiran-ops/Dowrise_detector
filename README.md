# Drowsiness Detector Pro

A Windows/Python 3.12 real-time computer-vision prototype for detecting prolonged eye closure and fatigue indicators.

## Features
- Real-time webcam monitoring
- MediaPipe Face Landmarker
- Eye Aspect Ratio (EAR)
- Sustained eye-closure detection
- Yawning indicator using Mouth Aspect Ratio (MAR)
- Lightweight head-pose/off-center indicator
- Drowsiness risk score
- Audible alarm
- Dashboard UI
- CSV event logging
- Reset/quit controls

## Run
```powershell
git pull
.\venv\Scripts\Activate.ps1
.\run.bat
```

The Face Landmarker model is downloaded automatically on first run.

## Controls
- **Q** — quit
- **R** — reset detector state

## Configuration
Tune thresholds in `config.py`. Webcam position and lighting affect EAR/MAR measurements.

Events are saved to `data/events.csv`.

> Prototype/assistive system only. It is not safety-certified and should not be the sole protection against driving while drowsy.
