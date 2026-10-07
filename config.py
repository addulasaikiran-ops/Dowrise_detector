"""Central configuration for the professional detector."""

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Drowsiness thresholds
EAR_THRESHOLD = 0.21
CLOSED_SECONDS = 1.2
WARNING_SECONDS = 0.55

# Yawning
MAR_THRESHOLD = 0.62
YAWN_SECONDS = 1.0

# Head pose heuristics
HEAD_TILT_THRESHOLD = 0.28
HEAD_POSE_SECONDS = 1.5

# Alarm
ALARM_BEEP_INTERVAL = 0.65
ALARM_FREQUENCY = 1000
ALARM_DURATION_MS = 180

WINDOW_TITLE = "Drowsiness Detector Pro"
LOG_FILE = "data/events.csv"
