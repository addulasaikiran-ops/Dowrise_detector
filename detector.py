"""Robust face-landmark metrics and temporal scoring helpers."""

import math
from collections import deque
from statistics import median

RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (263, 387, 385, 362, 380, 373)
MOUTH = (78, 308, 13, 14)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_aspect_ratio(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    horizontal = distance(p1, p4)
    if horizontal <= 1e-9:
        return 0.0
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks):
    left, right, top, bottom = [landmarks[i] for i in MOUTH]
    width = distance(left, right)
    return distance(top, bottom) / width if width > 1e-9 else 0.0


def eye_metrics(landmarks):
    return (
        eye_aspect_ratio(landmarks, RIGHT_EYE),
        eye_aspect_ratio(landmarks, LEFT_EYE),
    )


def blendshape_scores(categories):
    return {
        item.category_name: float(item.score)
        for item in (categories or [])
        if getattr(item, "category_name", None)
    }


def combined_eye_closure(landmarks, categories):
    right_ear, left_ear = eye_metrics(landmarks)
    ear = (right_ear + left_ear) / 2.0
    scores = blendshape_scores(categories)
    blink_left = scores.get("eyeBlinkLeft", 0.0)
    blink_right = scores.get("eyeBlinkRight", 0.0)
    blink = (blink_left + blink_right) / 2.0
    return ear, blink, right_ear, left_ear


def combined_mouth_openness(landmarks, categories):
    mar = mouth_aspect_ratio(landmarks)
    scores = blendshape_scores(categories)
    jaw_open = scores.get("jawOpen", 0.0)
    return mar, jaw_open


def head_pose_score(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]
    chin = landmarks[152]

    eye_mid_x = (left_eye.x + right_eye.x) / 2
    eye_span = max(distance(left_eye, right_eye), 1e-6)
    yaw = (nose.x - eye_mid_x) / eye_span

    eye_mid_y = (left_eye.y + right_eye.y) / 2
    face_height = max(abs(chin.y - eye_mid_y), 1e-6)
    pitch = (nose.y - eye_mid_y) / face_height
    return yaw, pitch


class MedianSmoother:
    def __init__(self, size=7):
        self.values = deque(maxlen=size)

    def reset(self):
        self.values.clear()

    def update(self, value):
        self.values.append(float(value))
        return float(median(self.values))


def drowsiness_score(
    ear,
    mar,
    eyes_closed,
    yawn,
    head_tilt,
    blink_score=0.0,
    baseline_ear=None,
):
    score = 0.0

    if eyes_closed:
        score += 72
    elif baseline_ear is not None and baseline_ear > 0:
        closure_ratio = max(0.0, 1.0 - (ear / baseline_ear))
        score += min(35.0, closure_ratio * 70.0)

    if yawn:
        score += 14

    if head_tilt:
        score += 14

    if blink_score > 0.55:
        score += 8

    if ear < 0.18:
        score += 8

    if mar > 0.75:
        score += 8

    return int(min(100, round(score)))


class RepeatedClosureTracker:
    """Track prolonged eye closures inside a rolling time window."""

    def __init__(self, window_seconds=30.0, minimum_seconds=0.70, count=3):
        self.window_seconds = float(window_seconds)
        self.minimum_seconds = float(minimum_seconds)
        self.required_count = int(count)
        self.timestamps = deque()

    def reset(self):
        self.timestamps.clear()

    def add(self, duration, now):
        if duration >= self.minimum_seconds:
            self.timestamps.append(float(now))
        self._prune(now)

    def _prune(self, now):
        cutoff = float(now) - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def triggered(self, now):
        self._prune(now)
        return len(self.timestamps) >= self.required_count


def alarm_condition(sustained_closure, repeated_closures, head_eye_combo, yawn_eye_combo):
    """Return True when any high-confidence drowsiness condition is present."""
    return (
        sustained_closure
        or repeated_closures
        or head_eye_combo
        or yawn_eye_combo
    )
