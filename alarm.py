import math
import time

import numpy as np
import pygame

from config import ALARM_BEEP_INTERVAL, ALARM_DURATION_MS, ALARM_FREQUENCY


class Alarm:
    def __init__(self):
        self.last_beep = 0.0
        pygame.mixer.init()
        sample_rate = 44100
        samples = int(sample_rate * ALARM_DURATION_MS / 1000)
        t = np.arange(samples) / sample_rate
        wave = (0.42 * np.sin(2 * math.pi * ALARM_FREQUENCY * t) * 32767).astype(
            np.int16
        )
        stereo = np.column_stack((wave, wave))
        self.sound = pygame.sndarray.make_sound(stereo)

    def update(self, active: bool) -> None:
        now = time.monotonic()
        if active and now - self.last_beep >= ALARM_BEEP_INTERVAL:
            self.sound.play()
            self.last_beep = now

    def stop(self) -> None:
        self.sound.stop()
        pygame.mixer.quit()
