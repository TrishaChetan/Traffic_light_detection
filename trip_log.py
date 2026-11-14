"""Daily trip log: every traffic-light change and every new piece of advice.

Rows go to logs/trip_YYYY-MM-DD.csv (one file per day, appended as you drive):

    date,time,event,label,distance_m,advice
    2026-11-14,08:31:02,light,red_light,12.5,
    2026-11-14,08:31:03,advice,red_light,12.5,Red light about 12.5 metres ahead. Brake now...

The dashboard links to today's file; /download_log?date=YYYY-MM-DD fetches older ones.
"""
import csv
import os
import re
import threading
from datetime import datetime

LOG_DIR = os.getenv("TRIP_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
FIELDS = ["date", "time", "event", "label", "distance_m", "advice"]
_lock = threading.Lock()


def log_path(day: str | None = None) -> str:
    day = day or datetime.now().strftime("%Y-%m-%d")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):  # also blocks path tricks like ../
        raise ValueError("date must be YYYY-MM-DD")
    return os.path.join(LOG_DIR, f"trip_{day}.csv")


def log_event(event: str, label, distance=None, advice: str = "") -> None:
    now = datetime.now()
    path = log_path(now.strftime("%Y-%m-%d"))
    row = {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"), "event": event,
           "label": label or "", "distance_m": "" if distance is None else distance, "advice": advice or ""}
    try:
        with _lock:
            os.makedirs(LOG_DIR, exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
    except OSError as e:  # logging must never stop detection
        print("⚠️ Could not write trip log:", e)


def available_days() -> list[str]:
    if not os.path.isdir(LOG_DIR):
        return []
    return sorted((m.group(1) for f in os.listdir(LOG_DIR)
                   if (m := re.fullmatch(r"trip_(\d{4}-\d{2}-\d{2})\.csv", f))), reverse=True)
