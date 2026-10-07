"""Face-landmark metrics used by the drowsiness detector."""

import math

RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (263, 387, 385, 362, 380, 373)
MOUTH = (78, 308, 13, 14)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_aspect_ratio(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    horizontal = distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks):
    left, right, top, bottom = [landmarks[i] for i in MOUTH]
    width = distance(left, right)
    return distance(top, bottom) / width if width else 0.0


def eye_metrics(landmarks):
    return eye_aspect_ratio(landmarks, RIGHT_EYE), eye_aspect_ratio(landmarks, LEFT_EYE)


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


def drowsiness_score(ear, mar, eyes_closed, yawn, head_tilt):
    score = 0
    if eyes_closed:
        score += 70
    if yawn:
        score += 15
    if head_tilt:
        score += 15
    if ear < 0.18:
        score = min(100, score + 10)
    if mar > 0.75:
        score = min(100, score + 10)
    return min(100, score)
