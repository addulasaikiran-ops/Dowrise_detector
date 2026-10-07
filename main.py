import os
import time
import urllib.request

import cv2
import mediapipe as mp

from alarm import Alarm
from config import (
    CAMERA_INDEX, CLOSED_SECONDS, EAR_THRESHOLD, FRAME_HEIGHT, FRAME_WIDTH,
    HEAD_POSE_SECONDS, HEAD_TILT_THRESHOLD, LOG_FILE, MAR_THRESHOLD,
    WINDOW_TITLE, WARNING_SECONDS, YAWN_SECONDS,
)
from detector import drowsiness_score, eye_metrics, head_pose_score, mouth_aspect_ratio
from logger import EventLogger

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe Face Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Face Landmarker model downloaded.")


def draw_bar(frame, label, value, x, y, width=240):
    value = max(0, min(100, int(value)))
    cv2.putText(frame, f"{label}: {value}%", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
    cv2.rectangle(frame, (x, y), (x + width, y + 16), (55, 55, 55), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * value / 100), y + 16),
                  (60, 190, 255), -1)


def main():
    ensure_model()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check Windows camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    Vision = mp.tasks.vision
    options = Vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=Vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    alarm = Alarm()
    logger = EventLogger(LOG_FILE)
    closed_since = yawn_since = head_since = None
    last_event = None
    frame_timestamp = 0

    try:
        with Vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_timestamp += 33
                result = landmarker.detect_for_video(image, frame_timestamp)

                status, ear, mar, score = "AWAKE", 0.0, 0.0, 0
                now = time.monotonic()

                if result.face_landmarks:
                    face = result.face_landmarks[0]
                    right_ear, left_ear = eye_metrics(face)
                    ear = (right_ear + left_ear) / 2
                    mar = mouth_aspect_ratio(face)
                    yaw, pitch = head_pose_score(face)

                    eyes_closed = ear < EAR_THRESHOLD
                    yawning = mar > MAR_THRESHOLD
                    head_tilt = abs(yaw) > HEAD_TILT_THRESHOLD or abs(pitch) > 0.42

                    closed_since = now if eyes_closed and closed_since is None else (closed_since if eyes_closed else None)
                    yawn_since = now if yawning and yawn_since is None else (yawn_since if yawning else None)
                    head_since = now if head_tilt and head_since is None else (head_since if head_tilt else None)

                    closed_duration = now - closed_since if closed_since else 0
                    yawn_duration = now - yawn_since if yawn_since else 0
                    head_duration = now - head_since if head_since else 0

                    sustained_closure = closed_duration >= CLOSED_SECONDS
                    sustained_yawn = yawn_duration >= YAWN_SECONDS
                    sustained_head = head_duration >= HEAD_POSE_SECONDS
                    score = drowsiness_score(ear, mar, sustained_closure, sustained_yawn, sustained_head)

                    if sustained_closure:
                        status = "DROWSY"
                    elif sustained_yawn:
                        status = "YAWNING"
                    elif sustained_head:
                        status = "HEAD OFF-CENTER"
                    elif closed_duration >= WARNING_SECONDS:
                        status = "WARNING"

                    if status != last_event and status in {"DROWSY", "WARNING", "YAWNING", "HEAD OFF-CENTER"}:
                        logger.log(status, ear, mar, score)
                        last_event = status
                else:
                    status = "NO FACE"
                    closed_since = yawn_since = head_since = None
                    last_event = None

                alarm.update(status == "DROWSY")

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (390, FRAME_HEIGHT), (18, 18, 18), -1)
                frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

                status_color = (0, 70, 255) if status == "DROWSY" else (0, 210, 120)
                cv2.putText(frame, "DROWSINESS DETECTOR PRO", (22, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                cv2.putText(frame, status, (22, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.05, status_color, 3)
                cv2.putText(frame, f"EAR  {ear:.3f}", (22, 124),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2)
                cv2.putText(frame, f"MAR  {mar:.3f}", (22, 156),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2)
                draw_bar(frame, "Drowsiness risk", score, 22, 192)

                cv2.putText(frame, "Q Quit   R Reset", (22, 282),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                if status == "DROWSY":
                    cv2.putText(frame, "WAKE UP!", (22, FRAME_HEIGHT - 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 70, 255), 3)

                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    closed_since = yawn_since = head_since = None
                    last_event = None
    finally:
        alarm.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
