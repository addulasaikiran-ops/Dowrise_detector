from unittest.mock import MagicMock, patch

import numpy as np

from alarm import Alarm


def test_alarm_initializes_and_plays_when_active():
    fake_sound = MagicMock()

    with patch("alarm.pygame.mixer.init"),          patch("alarm.pygame.sndarray.make_sound", return_value=fake_sound):
        alarm = Alarm()
        alarm.update(True)
        fake_sound.play.assert_called_once()
        alarm.stop()


def test_alarm_stops_immediately_when_inactive():
    fake_sound = MagicMock()

    with patch("alarm.pygame.mixer.init"),          patch("alarm.pygame.sndarray.make_sound", return_value=fake_sound):
        alarm = Alarm()
        alarm.update(True)
        alarm.update(False)
        assert fake_sound.stop.call_count >= 1
        alarm.stop()
