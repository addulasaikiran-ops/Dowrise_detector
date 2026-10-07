from types import SimpleNamespace

from detector import (
    MedianSmoother,
    RepeatedClosureTracker,
    combined_eye_closure,
    drowsiness_score,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)


def point(x, y):
    return SimpleNamespace(x=x, y=y)


def make_landmarks():
    pts = [point(0.5, 0.5) for _ in range(500)]

    # Right eye: horizontal = 0.20, vertical distances = 0.04 each.
    coords = {
        33: (0.30, 0.50), 160: (0.40, 0.46), 158: (0.40, 0.54),
        133: (0.50, 0.50), 153: (0.40, 0.46), 144: (0.40, 0.54),
        # Left eye
        263: (0.70, 0.50), 387: (0.60, 0.46), 385: (0.60, 0.54),
        362: (0.50, 0.50), 380: (0.60, 0.46), 373: (0.60, 0.54),
        # Mouth
        78: (0.40, 0.70), 308: (0.60, 0.70),
        13: (0.50, 0.67), 14: (0.50, 0.73),
    }
    for i, xy in coords.items():
        pts[i] = point(*xy)
    return pts


def test_eye_aspect_ratio_expected_value():
    landmarks = make_landmarks()
    assert abs(eye_aspect_ratio(landmarks, (33, 160, 158, 133, 153, 144)) - 0.4) < 1e-6


def test_mouth_aspect_ratio_expected_value():
    landmarks = make_landmarks()
    assert abs(mouth_aspect_ratio(landmarks) - 0.3) < 1e-6


def test_combined_eye_closure_uses_blink_blendshapes():
    class Category:
        def __init__(self, name, score):
            self.category_name = name
            self.score = score

    ear, blink, right, left = combined_eye_closure(
        make_landmarks(),
        [Category("eyeBlinkLeft", 0.8), Category("eyeBlinkRight", 0.6)],
    )
    assert abs(ear - 0.4) < 1e-6
    assert abs(blink - 0.7) < 1e-6
    assert right == left == 0.4


def test_median_smoother_rejects_single_spike():
    smoother = MedianSmoother(5)
    for value in [0.30, 0.31, 0.29, 0.95, 0.30]:
        result = smoother.update(value)
    assert result == 0.30


def test_repeated_closure_tracker_triggers_after_three_long_closures():
    tracker = RepeatedClosureTracker(window_seconds=30, minimum_seconds=0.7, count=3)
    tracker.add(0.8, 1.0)
    tracker.add(0.9, 5.0)
    assert not tracker.triggered(5.0)
    tracker.add(1.0, 10.0)
    assert tracker.triggered(10.0)


def test_repeated_closure_tracker_expires_old_events():
    tracker = RepeatedClosureTracker(window_seconds=10, minimum_seconds=0.7, count=3)
    tracker.add(0.8, 1.0)
    tracker.add(0.8, 2.0)
    tracker.add(0.8, 3.0)
    assert tracker.triggered(3.0)
    assert not tracker.triggered(20.0)


def test_short_closures_do_not_count():
    tracker = RepeatedClosureTracker(window_seconds=30, minimum_seconds=0.7, count=3)
    tracker.add(0.3, 1.0)
    tracker.add(0.5, 2.0)
    tracker.add(0.6, 3.0)
    assert not tracker.triggered(3.0)


def test_drowsiness_score_reaches_high_risk_for_sustained_closure():
    score = drowsiness_score(
        ear=0.14,
        mar=0.4,
        eyes_closed=True,
        yawn=False,
        head_tilt=False,
        blink_score=0.9,
        baseline_ear=0.30,
    )
    assert score >= 80


def test_normal_state_has_low_score():
    score = drowsiness_score(
        ear=0.30,
        mar=0.20,
        eyes_closed=False,
        yawn=False,
        head_tilt=False,
        blink_score=0.05,
        baseline_ear=0.30,
    )
    assert score < 10
