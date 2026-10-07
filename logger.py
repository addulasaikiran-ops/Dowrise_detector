import csv
import os
from datetime import datetime


class EventLogger:
    def __init__(self, path):
        self.path = path
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "event", "ear", "mar", "score"])

    def log(self, event, ear, mar, score):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [datetime.now().isoformat(timespec="seconds"), event,
                 f"{ear:.3f}", f"{mar:.3f}", score]
            )
