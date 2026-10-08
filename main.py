import os
import time
import urllib.request
from statistics import median

import cv2
import mediapipe as mp

from alarm import Alarm
from config import (
    BLINK_CLOSED_THRESHOLD,
    CALIBRATION_SECONDS,
    CAMERA_INDEX,
    CLOSED_SECONDS,
    EAR_CLOSED_RATIO,
    EAR_THRESHOLD,
    EAR_WARNING_RATIO,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HEAD_PITCH_THRESHOLD,
    HEAD_POSE_SECONDS,
    HEAD_TILT_THRESHOLD,
    JAW_OPEN_THRESHOLD,
    LOG_FILE,
    MAR_THRESHOLD,
    MIN_CALIBRATION_SAMPLES,
    REPEATED_CLOSURE_COUNT,
    REPEATED_CLOSURE_MIN_SECONDS,
    REPEATED_CLOSURE_WINDOW,
    HEAD_EYE_COMBO_SECONDS,
    YAWN_EYE_COMBO_SECONDS,
    SMOOTHING_WINDOW,
    WARNING_SECONDS,
    YAWN_SECONDS,
)
from detector import (
    MedianSmoother,
    calibration_baseline,
    eye_thresholds,
    RepeatedClosureTracker,
    alarm_condition,
    combined_eye_closure,
    combined_mouth_openness,
    drowsiness_score,
    head_pose_score,
)
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


def analyze_photo(path):
    """Analyze one still image without opening the webcam."""
    ensure_model()

    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("The selected image could not be read.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    Vision = mp.tasks.vision
    options = Vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=Vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        output_face_blendshapes=True,
    )

    try:
        with Vision.FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(image)
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe could not process this image. "
            "Check that the face is clearly visible and try another photo."
        ) from exc

    if not result.face_landmarks:
        return frame, "NO FACE", 0.0, 0.0, 0

    face = result.face_landmarks[0]
    categories = result.face_blendshapes[0] if result.face_blendshapes else []

    ear, blink, _, _ = combined_eye_closure(face, categories)
    mar, jaw_open = combined_mouth_openness(face, categories)

    eyes_closed = ear < EAR_THRESHOLD and blink >= BLINK_CLOSED_THRESHOLD
    yawn = mar > MAR_THRESHOLD and jaw_open >= JAW_OPEN_THRESHOLD

    # A single image cannot prove sustained drowsiness. Report visible state.
    if eyes_closed:
        status = "EYES CLOSED"
        score = min(
            65,
            drowsiness_score(
                ear, mar, False, yawn, False, blink_score=blink
            ),
        )
    elif yawn:
        status = "YAWNING"
        score = drowsiness_score(
            ear, mar, False, True, False, blink_score=blink
        )
    else:
        status = "AWAKE"
        score = drowsiness_score(
            ear, mar, False, False, False, blink_score=blink
        )

    return frame, status, ear, mar, score


class DetectionEngine:
    def __init__(self):
        ensure_model()

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check Windows camera permissions."
            )

        Vision = mp.tasks.vision
        options = Vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=Vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.55,
            min_face_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_face_blendshapes=True,
        )

        self.landmarker = Vision.FaceLandmarker.create_from_options(options)
        self.alarm = Alarm()
        self.logger = EventLogger(LOG_FILE)

        self.frame_timestamp = 0
        self.timestamp_start = time.monotonic()
        self.reset()

    def reset(self):
        self.closed_since = None
        self.yawn_since = None
        self.head_since = None
        self.last_event = None
        self.drowsy_latched = False
        self.recent_closures = RepeatedClosureTracker(
            REPEATED_CLOSURE_WINDOW,
            REPEATED_CLOSURE_MIN_SECONDS,
            REPEATED_CLOSURE_COUNT,
        )

        self.ear_smoother = MedianSmoother(SMOOTHING_WINDOW)
        self.mar_smoother = MedianSmoother(SMOOTHING_WINDOW)
        self.blink_smoother = MedianSmoother(SMOOTHING_WINDOW)
        self.jaw_smoother = MedianSmoother(SMOOTHING_WINDOW)
        self.yaw_smoother = MedianSmoother(SMOOTHING_WINDOW)
        self.pitch_smoother = MedianSmoother(SMOOTHING_WINDOW)

        self.calibration_start = None
        self.calibration_samples = []
        self.baseline_ear = None
        self.calibrated = False
        self.recent_closures.reset()

    def _update_calibration(self, now, ear, blink):
        if self.calibrated:
            return

        if self.calibration_start is None:
            self.calibration_start = now

        elapsed = now - self.calibration_start

        if blink < 0.35 and ear > 0.18:
            self.calibration_samples.append(ear)

        if (
            elapsed >= CALIBRATION_SECONDS
            and len(self.calibration_samples) >= MIN_CALIBRATION_SAMPLES
        ):
            self.baseline_ear = calibration_baseline(
                self.calibration_samples, MIN_CALIBRATION_SAMPLES
            )
            self.calibrated = self.baseline_ear is not None

    def _eye_thresholds(self):
        return eye_thresholds(
            self.baseline_ear,
            EAR_THRESHOLD,
            EAR_CLOSED_RATIO,
            EAR_WARNING_RATIO,
        )

    def process(self):
        ok, frame = self.cap.read()
        if not ok:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self.timestamp_start) * 1000)
        if timestamp_ms <= self.frame_timestamp:
            timestamp_ms = self.frame_timestamp + 1
        self.frame_timestamp = timestamp_ms

        result = self.landmarker.detect_for_video(image, timestamp_ms)

        now = time.monotonic()
        status = "AWAKE"
        ear = 0.0
        mar = 0.0
        score = 0

        if result.face_landmarks:
            face = result.face_landmarks[0]
            categories = result.face_blendshapes[0] if result.face_blendshapes else []

            raw_ear, raw_blink, _, _ = combined_eye_closure(face, categories)
            raw_mar, raw_jaw = combined_mouth_openness(face, categories)
            raw_yaw, raw_pitch = head_pose_score(face)

            ear = self.ear_smoother.update(raw_ear)
            mar = self.mar_smoother.update(raw_mar)
            blink = self.blink_smoother.update(raw_blink)
            jaw_open = self.jaw_smoother.update(raw_jaw)
            yaw = self.yaw_smoother.update(raw_yaw)
            pitch = self.pitch_smoother.update(raw_pitch)

            self._update_calibration(now, ear, blink)
            closed_threshold, warning_threshold = self._eye_thresholds()

            geometry_closed = ear < closed_threshold
            blendshape_closed = blink >= BLINK_CLOSED_THRESHOLD
            eyes_closed = geometry_closed and blendshape_closed

            if not self.calibrated:
                eyes_closed = ear < EAR_THRESHOLD and blendshape_closed

            yawning = mar > MAR_THRESHOLD and jaw_open >= JAW_OPEN_THRESHOLD
            head_tilt = (
                abs(yaw) > HEAD_TILT_THRESHOLD
                or abs(pitch) > HEAD_PITCH_THRESHOLD
            )

            if eyes_closed:
                if self.closed_since is None:
                    self.closed_since = now
            else:
                if self.closed_since is not None:
                    self.recent_closures.add(now - self.closed_since, now)
                self.closed_since = None

            if yawning:
                if self.yawn_since is None:
                    self.yawn_since = now
            else:
                self.yawn_since = None

            if head_tilt:
                if self.head_since is None:
                    self.head_since = now
            else:
                self.head_since = None

            closed_duration = now - self.closed_since if self.closed_since else 0.0
            yawn_duration = now - self.yawn_since if self.yawn_since else 0.0
            head_duration = now - self.head_since if self.head_since else 0.0

            sustained_closure = closed_duration >= CLOSED_SECONDS
            sustained_yawn = yawn_duration >= YAWN_SECONDS
            sustained_head = head_duration >= HEAD_POSE_SECONDS

            repeated_closures = self.recent_closures.triggered(now)

            head_eye_combo = (
                eyes_closed
                and closed_duration >= HEAD_EYE_COMBO_SECONDS
                and sustained_head
            )
            yawn_eye_combo = (
                sustained_yawn
                and blink >= BLINK_CLOSED_THRESHOLD
                and yawn_duration >= YAWN_EYE_COMBO_SECONDS
            )

            alarm_triggered = alarm_condition(
                sustained_closure,
                repeated_closures,
                head_eye_combo,
                yawn_eye_combo,
            )

            score = drowsiness_score(
                ear,
                mar,
                sustained_closure,
                sustained_yawn,
                sustained_head,
                blink_score=blink,
                baseline_ear=self.baseline_ear,
            )

            if alarm_triggered:
                self.drowsy_latched = True
            elif self.drowsy_latched:
                if (
                    not eyes_closed
                    and blink < 0.40
                    and ear >= warning_threshold
                ):
                    self.drowsy_latched = False

            if self.drowsy_latched:
                status = "DROWSY"
            elif sustained_yawn:
                status = "YAWNING"
            elif sustained_head:
                status = "HEAD OFF-CENTER"
            elif closed_duration >= WARNING_SECONDS:
                status = "WARNING"

            if status != self.last_event and status in {
                "DROWSY",
                "WARNING",
                "YAWNING",
                "HEAD OFF-CENTER",
            }:
                self.logger.log(status, ear, mar, score)
                self.last_event = status

        else:
            status = "NO FACE"
            self.closed_since = None
            self.yawn_since = None
            self.head_since = None
            self.drowsy_latched = False
            self.last_event = None

        self.alarm.update(status == "DROWSY")
        return frame, status, ear, mar, score

    def close(self):
        self.alarm.stop()
        self.landmarker.close()
        self.cap.release()
