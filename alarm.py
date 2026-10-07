import math
import time

import numpy as np
import pygame

from config import (
    ALARM_BEEP_INTERVAL,
    ALARM_DURATION_MS,
    ALARM_FREQUENCY,
    ALARM_VOLUME,
)


class Alarm:
    """Urgent repeating drowsiness alarm with immediate stop on recovery."""

    def __init__(self):
        self.last_beep = 0.0
        self.active = False
        pygame.mixer.init(frequency=44100, size=-16, channels=2)

        sample_rate = 44100
        samples = int(sample_rate * ALARM_DURATION_MS / 1000)
        t = np.arange(samples) / sample_rate

        # Two-tone sweep gives the alarm a more noticeable automotive warning
        # character than a flat continuous beep.
        tone1 = np.sin(2 * math.pi * ALARM_FREQUENCY * t)
        tone2 = np.sin(2 * math.pi * (ALARM_FREQUENCY * 1.32) * t)
        envelope = np.minimum(1.0, np.arange(samples) / (sample_rate * 0.012))
        envelope *= np.minimum(1.0, (samples - np.arange(samples)) / (sample_rate * 0.025))
        wave = (ALARM_VOLUME * 0.5 * (tone1 + tone2) * envelope * 32767).astype(np.int16)

        stereo = np.column_stack((wave, wave))
        self.sound = pygame.sndarray.make_sound(stereo)

    def update(self, active: bool) -> None:
        now = time.monotonic()

        if not active:
            if self.active:
                self.sound.stop()
            self.active = False
            self.last_beep = 0.0
            return

        self.active = True

        if now - self.last_beep >= ALARM_BEEP_INTERVAL:
            self.sound.play()
            self.last_beep = now

    def stop(self) -> None:
        self.sound.stop()
        pygame.mixer.quit()
        self.active = False
