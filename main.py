import os
import time
import urllib.request

import cv2
import mediapipe as mp

from alarm import Alarm
from config import (
    CAMERA_INDEX,
    CLOSED_FRAMES_THRESHOLD,
    EAR_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    WINDOW_TITLE,
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Face Landmarker indexes for the eye contours.
# These indexes are from MediaPipe's face landmark topology.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (263, 387, 385, 362, 380, 373)


def ensure_model() -> None:
    """Download the Face Landmarker model once if it is not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        return

    print("Downloading MediaPipe Face Landmarker model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise RuntimeError(
            "Could not download the MediaPipe model. Check your internet connection."
        ) from exc

    print("Face Landmarker model downloaded.")


def euclidean_distance(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def eye_aspect_ratio(landmarks, eye) -> float:
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def main() -> None:
    ensure_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions and CAMERA_INDEX."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    alarm = Alarm()
    closed_frames = 0
    fps_time = time.monotonic()
    fps = 0.0
    frame_timestamp_ms = 0

    with FaceLandmarker.create_from_options(options) as face_landmarker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                frame_timestamp_ms += 33
                result = face_landmarker.detect_for_video(
                    mp_image, frame_timestamp_ms
                )

                status = "AWAKE"
                ear = None
                drowsy = False

                if result.face_landmarks:
                    face = result.face_landmarks[0]
                    right_ear = eye_aspect_ratio(face, RIGHT_EYE)
                    left_ear = eye_aspect_ratio(face, LEFT_EYE)
                    ear = (right_ear + left_ear) / 2.0

                    if ear < EAR_THRESHOLD:
                        closed_frames += 1
                    else:
                        closed_frames = max(0, closed_frames - 2)

                    if closed_frames >= CLOSED_FRAMES_THRESHOLD:
                        drowsy = True
                        status = "DROWSY"
                    elif closed_frames > 0:
                        status = "EYES CLOSING"
                else:
                    closed_frames = max(0, closed_frames - 1)
                    status = "NO FACE"

                alarm.update(drowsy)

                current = time.monotonic()
                elapsed = current - fps_time
                if elapsed >= 1.0:
                    fps = 1.0 / max(elapsed, 1e-6)
                    fps_time = current

                cv2.putText(
                    frame,
                    f"Status: {status}",
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255) if drowsy else (0, 200, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}" if ear is not None else "EAR: --",
                    (25, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Closed frames: {closed_frames}",
                    (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (25, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if drowsy:
                    cv2.putText(
                        frame,
                        "!!! WAKE UP !!!",
                        (25, FRAME_HEIGHT - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                    )

                cv2.imshow(WINDOW_TITLE, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            alarm.stop()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
