from detector import alarm_condition


def test_sustained_closure_triggers_alarm():
    assert alarm_condition(True, False, False, False)


def test_repeated_closures_trigger_alarm():
    assert alarm_condition(False, True, False, False)


def test_head_and_eye_combination_triggers_alarm():
    assert alarm_condition(False, False, True, False)


def test_yawn_and_eye_combination_triggers_alarm():
    assert alarm_condition(False, False, False, True)


def test_no_drowsiness_signal_does_not_trigger_alarm():
    assert not alarm_condition(False, False, False, False)
