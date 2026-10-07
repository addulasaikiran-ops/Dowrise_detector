import os
import time
import urllib.request

import cv2
import mediapipe as mp

from alarm import Alarm
from config import (
    CAMERA_INDEX, CLOSED_SECONDS, EAR_THRESHOLD, FRAME_HEIGHT, FRAME_WIDTH,
    HEAD_POSE_SECONDS, HEAD_TILT_THRESHOLD, LOG_FILE, MAR_THRESHOLD,
    WARNING_SECONDS, YAWN_SECONDS,
)
from detector import drowsiness_score, eye_metrics, head_pose_score, mouth_aspect_ratio
from logger import EventLogger

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe Face Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Face Landmarker model downloaded.")


class DetectionEngine:
    def __init__(self):
        ensure_model()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check Windows camera permissions.")

        Vision = mp.tasks.vision
        options = Vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=Vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = Vision.FaceLandmarker.create_from_options(options)
        self.alarm = Alarm()
        self.logger = EventLogger(LOG_FILE)
        self.reset()
        self.frame_timestamp = 0

    def reset(self):
        self.closed_since = None
        self.yawn_since = None
        self.head_since = None
        self.last_event = None

    def process(self):
        ok, frame = self.cap.read()
        if not ok:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_timestamp += 33
        result = self.landmarker.detect_for_video(image, self.frame_timestamp)

        now = time.monotonic()
        status, ear, mar, score = "AWAKE", 0.0, 0.0, 0

        if result.face_landmarks:
            face = result.face_landmarks[0]
            right_ear, left_ear = eye_metrics(face)
            ear = (right_ear + left_ear) / 2
            mar = mouth_aspect_ratio(face)
            yaw, pitch = head_pose_score(face)

            eyes_closed = ear < EAR_THRESHOLD
            yawning = mar > MAR_THRESHOLD
            head_tilt = abs(yaw) > HEAD_TILT_THRESHOLD or abs(pitch) > 0.42

            self.closed_since = now if eyes_closed and self.closed_since is None else (self.closed_since if eyes_closed else None)
            self.yawn_since = now if yawning and self.yawn_since is None else (self.yawn_since if yawning else None)
            self.head_since = now if head_tilt and self.head_since is None else (self.head_since if head_tilt else None)

            closed_duration = now - self.closed_since if self.closed_since else 0
            yawn_duration = now - self.yawn_since if self.yawn_since else 0
            head_duration = now - self.head_since if self.head_since else 0

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

            if status != self.last_event and status in {"DROWSY", "WARNING", "YAWNING", "HEAD OFF-CENTER"}:
                self.logger.log(status, ear, mar, score)
                self.last_event = status
        else:
            status = "NO FACE"
            self.closed_since = self.yawn_since = self.head_since = None
            self.last_event = None

        self.alarm.update(status == "DROWSY")
        return frame, status, ear, mar, score

    def close(self):
        self.alarm.stop()
        self.landmarker.close()
        self.cap.release()
