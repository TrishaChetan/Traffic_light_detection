"""Save annotated frames: on demand from the dashboard, or automatically when a
red (or any) light appears.

    SNAPSHOT_DIR=snapshots      where the JPEGs go
    SNAPSHOT_KEEP=200           oldest files are deleted beyond this many
    AUTO_SNAPSHOT=red           "red", "change" (every light change) or "off" (default)

Each image gets a caption strip burned in (time, light, distance), so a single
file is self-explanatory when shared.
"""
import os
import re
import time

import cv2

SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
SNAPSHOT_KEEP = int(os.getenv("SNAPSHOT_KEEP", "200"))
AUTO_SNAPSHOT = os.getenv("AUTO_SNAPSHOT", "off").strip().lower()
NAME_RE = re.compile(r"^\d{8}-\d{6}(-\d+)?-[a-z_]+-(manual|auto)\.jpg$")


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z]+", "_", (label or "none").lower()).strip("_")
    return s[:30] or "none"


def caption(frame, label: str, distance, when: float):
    """Return a copy of the frame with a dark caption strip at the bottom."""
    out = frame.copy()
    h, w = out.shape[:2]
    strip = max(28, h // 14)
    cv2.rectangle(out, (0, h - strip), (w, h), (0, 0, 0), -1)
    text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when)) + f"  |  {label or 'no light'}"
    if distance is not None:
        text += f"  |  ~{distance} m"
    scale = max(0.45, strip / 48)
    cv2.putText(out, text, (8, h - strip // 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save(frame, label: str, distance=None, reason: str = "manual", when: float | None = None,
         directory: str | None = None) -> str:
    """Write a captioned JPEG and return its file name."""
    directory = directory or SNAPSHOT_DIR
    when = when or time.time()
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when))
    name = f"{stamp}-{_slug(label)}-{reason}.jpg"
    n = 1
    while os.path.exists(os.path.join(directory, name)):  # two in the same second
        n += 1
        name = f"{stamp}-{n}-{_slug(label)}-{reason}.jpg"
    if not cv2.imwrite(os.path.join(directory, name), caption(frame, label, distance, when)):
        raise OSError(f"could not write snapshot to {directory}")
    prune(directory)
    return name


def list_snapshots(directory: str | None = None, limit: int = 50) -> list[dict]:
    directory = directory or SNAPSHOT_DIR
    if not os.path.isdir(directory):
        return []
    names = sorted((f for f in os.listdir(directory) if NAME_RE.match(f)), reverse=True)
    out = []
    for f in names[:limit]:
        parts = f[:-4].split("-")
        out.append({"name": f, "label": parts[-2].replace("_", " "), "reason": parts[-1],
                    "time": f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]} {parts[1][:2]}:{parts[1][2:4]}:{parts[1][4:]}"})
    return out


def path_for(name: str, directory: str | None = None) -> str | None:
    """Absolute path of a snapshot, or None for anything that isn't one (no traversal)."""
    if not NAME_RE.match(name or ""):
        return None
    p = os.path.join(os.path.abspath(directory or SNAPSHOT_DIR), name)
    return p if os.path.isfile(p) else None


def prune(directory: str | None = None, keep: int | None = None):
    directory = directory or SNAPSHOT_DIR
    keep = SNAPSHOT_KEEP if keep is None else keep
    names = sorted(f for f in os.listdir(directory) if NAME_RE.match(f))
    for f in names[:max(0, len(names) - keep)]:
        os.remove(os.path.join(directory, f))


def should_auto_snapshot(new_label: str, mode: str | None = None) -> bool:
    mode = (mode or AUTO_SNAPSHOT)
    if mode == "red":
        return "red" in (new_label or "").lower()
    if mode == "change":
        return bool(new_label) and "no traffic light" not in new_label.lower()
    return False
