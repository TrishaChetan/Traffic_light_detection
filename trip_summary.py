"""Summarise a day's trip log: lights seen, time waited at reds, camera dropouts.

    python trip_summary.py              # today
    python trip_summary.py 2026-11-14   # an earlier day

Also served as JSON at /trip_summary (?date=YYYY-MM-DD) and shown on the dashboard.

A "red wait" is the time from a red light appearing to the next green light,
ignoring gaps longer than MAX_WAIT_S (the car most likely drove off and the
camera just missed the green).
"""
import csv
import os
import sys
from datetime import datetime

import trip_log

MAX_WAIT_S = 300
COLOURS = ("red", "yellow", "green")


def _colour(label: str) -> str | None:
    label = (label or "").lower()
    return next((c for c in COLOURS if c in label), None)


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def summarize_rows(rows: list[dict]) -> dict:
    lights = {c: 0 for c in COLOURS}
    waits, advice, outages = [], 0, 0
    red_since = None
    first = last = None
    for row in rows:
        try:
            ts = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            continue
        first = first or ts
        last = ts
        event = row.get("event")
        if event == "advice":
            advice += 1
        elif event == "camera_offline":
            outages += 1
            red_since = None
        elif event == "light":
            colour = _colour(row.get("label"))
            if colour:
                lights[colour] += 1
            if colour == "red":
                red_since = red_since or ts
            elif colour == "green" and red_since:
                wait = (ts - red_since).total_seconds()
                if 0 <= wait <= MAX_WAIT_S:
                    waits.append(wait)
                red_since = None
            elif colour is None:  # light left the view
                pass
    return {
        "lights": lights,
        "total_lights": sum(lights.values()),
        "red_waits": len(waits),
        "avg_red_wait_s": round(sum(waits) / len(waits), 1) if waits else None,
        "longest_red_wait_s": round(max(waits), 1) if waits else None,
        "total_red_wait_s": round(sum(waits), 1),
        "advice_given": advice,
        "camera_dropouts": outages,
        "first_event": first.strftime("%H:%M:%S") if first else None,
        "last_event": last.strftime("%H:%M:%S") if last else None,
    }


def summarize(day: str | None = None) -> dict:
    path = trip_log.log_path(day)  # validates the date
    day = os.path.basename(path)[5:15]
    if not os.path.exists(path):
        return {"date": day, "found": False, **summarize_rows([])}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {"date": day, "found": True, **summarize_rows(rows)}


def report(s: dict) -> str:
    if not s["found"]:
        return f"No trip log for {s['date']} yet."
    l = s["lights"]
    return "\n".join([
        f"Trip summary — {s['date']} ({s['first_event']} to {s['last_event']})",
        f"  Lights seen:   {s['total_lights']}  (red {l['red']}, yellow {l['yellow']}, green {l['green']})",
        f"  Red waits:     {s['red_waits']}  · average {_fmt(s['avg_red_wait_s'])}"
        f" · longest {_fmt(s['longest_red_wait_s'])} · total {_fmt(s['total_red_wait_s'])}",
        f"  Advice spoken: {s['advice_given']}",
        f"  Camera drops:  {s['camera_dropouts']}",
    ])


if __name__ == "__main__":
    try:
        print(report(summarize(sys.argv[1] if len(sys.argv) > 1 else None)))
    except ValueError as e:
        sys.exit(str(e))
