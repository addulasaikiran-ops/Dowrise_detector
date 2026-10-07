import pytest

from config import EAR_CLOSED_RATIO, EAR_THRESHOLD
from main import DetectionEngine


def test_eye_thresholds_use_personal_baseline():
    engine = DetectionEngine.__new__(DetectionEngine)
    engine.baseline_ear = 0.30
    closed, warning = engine._eye_thresholds()

    assert closed == pytest.approx(min(EAR_THRESHOLD, 0.30 * EAR_CLOSED_RATIO))
    assert warning > closed


def test_eye_thresholds_have_safe_fallback_before_calibration():
    engine = DetectionEngine.__new__(DetectionEngine)
    engine.baseline_ear = None
    closed, warning = engine._eye_thresholds()

    assert closed == EAR_THRESHOLD
    assert warning == pytest.approx(EAR_THRESHOLD * 1.10)


def test_calibration_ignores_blink_samples():
    engine = DetectionEngine.__new__(DetectionEngine)
    engine.calibration_start = None
    engine.calibration_samples = []
    engine.baseline_ear = None
    engine.calibrated = False

    engine._update_calibration(0.0, 0.30, 0.8)
    assert engine.calibration_samples == []


def test_calibration_learns_open_eye_baseline():
    engine = DetectionEngine.__new__(DetectionEngine)
    engine.calibration_start = None
    engine.calibration_samples = []
    engine.baseline_ear = None
    engine.calibrated = False

    for i in range(31):
        engine._update_calibration(0.1 * i, 0.30, 0.10)

    assert engine.calibrated
    assert engine.baseline_ear == pytest.approx(0.30)
