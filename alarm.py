import math
import time

import numpy as np
import pygame


class Alarm:
    """Simple repeating beep alarm using pygame."""

    def __init__(self, frequency=880, duration_ms=180):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.enabled = False
        self.last_beep = 0.0
        self.beep_interval = 0.8

        pygame.mixer.init()

        sample_rate = 44100
        samples = int(sample_rate * duration_ms / 1000)
        t = np.arange(samples) / sample_rate
        wave = (0.4 * np.sin(2 * math.pi * frequency * t) * 32767).astype(np.int16)
        stereo = np.column_stack((wave, wave))
        self.sound = pygame.sndarray.make_sound(stereo)

    def update(self, active: bool) -> None:
        now = time.monotonic()

        if active:
            self.enabled = True
            if now - self.last_beep >= self.beep_interval:
                self.sound.play()
                self.last_beep = now
        else:
            self.enabled = False

    def stop(self) -> None:
        self.sound.stop()
        pygame.mixer.quit()
