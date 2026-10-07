def test_drowsiness_alarm_rule_combinations():
    def alarm_condition(sustained, repeated, head_eye, yawn_eye):
        return sustained or repeated or head_eye or yawn_eye

    assert alarm_condition(True, False, False, False)
    assert alarm_condition(False, True, False, False)
    assert alarm_condition(False, False, True, False)
    assert alarm_condition(False, False, False, True)
    assert not alarm_condition(False, False, False, False)
