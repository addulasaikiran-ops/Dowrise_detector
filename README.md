# Drowsiness Detector

A real-time Python prototype that uses a webcam to detect prolonged eye closure and trigger an alarm.

## Stack
- Python 3.11
- OpenCV
- MediaPipe Face Mesh
- NumPy
- Pygame

## Windows setup

1. Open this repository folder in VS Code.
2. Open a PowerShell terminal.
3. Run:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then start the detector:

```powershell
python main.py
```

Press **Q** to quit.

## What it does

The prototype tracks one face, calculates an Eye Aspect Ratio (EAR) from eye landmarks, and treats sustained low EAR as prolonged eye closure. When the closure lasts long enough, the status changes to **DROWSY** and an audible alarm starts.

The default thresholds live in `config.py` and can be tuned for your camera and lighting.

> This is a prototype/assistive computer-vision project and is not a safety-certified driver monitoring system.
