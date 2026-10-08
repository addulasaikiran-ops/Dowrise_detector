import pytest

from config import EAR_CLOSED_RATIO, EAR_THRESHOLD, MIN_CALIBRATION_SAMPLES
from detector import calibration_baseline, eye_thresholds


def test_eye_thresholds_use_personal_baseline():
    closed, warning = eye_thresholds(
        0.30,
        EAR_THRESHOLD,
        EAR_CLOSED_RATIO,
        0.84,
    )

    assert closed == pytest.approx(min(EAR_THRESHOLD, 0.30 * EAR_CLOSED_RATIO))
    assert warning > closed


def test_eye_thresholds_have_safe_fallback_before_calibration():
    closed, warning = eye_thresholds(
        None,
        EAR_THRESHOLD,
        EAR_CLOSED_RATIO,
        0.84,
    )

    assert closed == EAR_THRESHOLD
    assert warning == pytest.approx(EAR_THRESHOLD * 1.10)


def test_calibration_requires_minimum_samples():
    assert calibration_baseline([0.30] * (MIN_CALIBRATION_SAMPLES - 1), MIN_CALIBRATION_SAMPLES) is None


def test_calibration_learns_open_eye_baseline():
    baseline = calibration_baseline([0.30] * MIN_CALIBRATION_SAMPLES, MIN_CALIBRATION_SAMPLES)
    assert baseline == pytest.approx(0.30)
