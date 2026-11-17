import os
import re
import threading
import time

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# =====================
# CONFIG
# =====================
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
# CAMERA_SOURCE can be a stream URL (e.g. IP Webcam app: http://<phone-ip>:8080/video)
# or a webcam index such as 0. Set it in .env.
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "http://192.168.31.37:8080/video")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.4"))
IMG_SIZE = 320
K = 2000
RECONNECT_DELAY = 2  # seconds between camera reconnect attempts
BOX_COLORS = {"red": (0, 0, 255), "yellow": (0, 215, 255), "green": (0, 200, 0)}

# =====================
# LOAD MODEL
# =====================
print("🚀 Loading YOLO model...")
model = YOLO(MODEL_PATH)
model.fuse()
print("✅ Model loaded and fused")

# =====================
# GLOBALS
# =====================
latest_camera_frame = None
camera_lock = threading.Lock()

# If no new frame arrives for this long, the camera is treated as offline and
# detection stops — otherwise the last frozen frame would keep "showing" a light.
STALE_AFTER = float(os.getenv("STALE_AFTER", "2.0"))

stats = {
    "camera_online": False,
    "last_frame_at": 0.0,
    "camera_reconnects": 0,
    "detection_fps": 0.0,
    "inference_ms": None,
}
stats_lock = threading.Lock()


def _masked_source(src: str) -> str:
    # hide credentials in URLs like http://user:pass@192.168.1.20:8080/video
    return re.sub(r"//[^/@]+@", "//***@", src)


def get_status() -> dict:
    """Snapshot for /health and the dashboard."""
    with stats_lock:
        s = dict(stats)
    age = time.time() - s["last_frame_at"] if s["last_frame_at"] else None
    s["camera_online"] = bool(age is not None and age < STALE_AFTER)
    s["frame_age_s"] = None if age is None else round(age, 2)
    s["camera_source"] = _masked_source(CAMERA_SOURCE)
    s["model"] = os.path.basename(MODEL_PATH)
    return s

latest_detection = {
    "frame": None,
    "label": "No traffic light detected",
    "distance": None,
    "last_seen": 0
}
detection_lock = threading.Lock()

DETECTION_HOLD_TIME = 3  # seconds


# =====================
# CAMERA THREAD
# =====================
def open_camera():
    source = CAMERA_SOURCE.strip()
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))          # local webcam
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)  # network stream
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def camera_loop():
    """Read frames forever, reconnecting if the camera/stream drops."""
    global latest_camera_frame

    print(f"📷 Camera thread started (source: {CAMERA_SOURCE})")
    while True:
        cap = open_camera()
        if not cap.isOpened():
            with stats_lock:
                stats["camera_online"] = False
            print(f"❌ Camera failed to open — retrying in {RECONNECT_DELAY}s")
            time.sleep(RECONNECT_DELAY)
            continue

        failures = 0
        while failures < 50:  # ~0.5 s of consecutive failed reads = stream lost
            ret, frame = cap.read()
            if not ret or frame is None:
                failures += 1
                time.sleep(0.01)
                continue
            failures = 0
            with camera_lock:
                latest_camera_frame = frame
            with stats_lock:
                stats["last_frame_at"] = time.time()
                stats["camera_online"] = True
            time.sleep(0.01)

        print("⚠️ Camera stream lost — reconnecting")
        with stats_lock:
            stats["camera_online"] = False
            stats["camera_reconnects"] += 1
        cap.release()
        time.sleep(RECONNECT_DELAY)


def draw_detection(frame, box_xyxy, label, conf, distance):
    """Draw the detection (given in IMG_SIZE coordinates) on the full-size frame."""
    h, w = frame.shape[:2]
    sx, sy = w / IMG_SIZE, h / IMG_SIZE
    x1, y1, x2, y2 = box_xyxy
    p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
    color = next((c for k, c in BOX_COLORS.items() if k in label.lower()), (255, 255, 255))
    cv2.rectangle(frame, p1, p2, color, 2)
    text = f"{label} {conf:.0%}" + (f" ~{distance} m" if distance else "")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(p1[1], th + 8)
    cv2.rectangle(frame, (p1[0], ty - th - 8), (p1[0] + tw + 6, ty), color, -1)
    cv2.putText(frame, text, (p1[0] + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


# =====================
# DETECTION THREAD
# =====================
def detection_loop():
    global latest_camera_frame, latest_detection

    print("🟢 Detection loop started")

    while True:
        with camera_lock:
            frame = None if latest_camera_frame is None else latest_camera_frame.copy()
        with stats_lock:
            last_frame_at = stats["last_frame_at"]
        if frame is None:
            time.sleep(0.05)
            continue

        # Camera went quiet: don't keep re-detecting a frozen frame.
        if time.time() - last_frame_at > STALE_AFTER:
            with detection_lock:
                latest_detection["label"] = "No traffic light detected"
                latest_detection["distance"] = None
                latest_detection["frame"] = None
            with stats_lock:
                stats["detection_fps"] = 0.0
            time.sleep(0.2)
            continue

        label = None
        distance = None
        t0 = time.time()

        try:
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            results = model(resized, conf=CONF_THRESHOLD, verbose=False)
            infer_ms = (time.time() - t0) * 1000

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                # use the most confident detection, not just the first one
                best = int(boxes.conf.argmax().item())
                box = boxes[best]

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = model.names[cls_id]

                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                box_width = xmax - xmin

                if box_width > 0:
                    distance = round(K / box_width, 1)

                frame = draw_detection(frame, (xmin, ymin, xmax, ymax), label, conf, distance)

        except Exception as e:
            print("⚠️ Detection error:", e)
            infer_ms = None

        current_time = time.time()
        with stats_lock:
            loop_s = max(current_time - t0 + 0.05, 1e-3)
            # smoothed frames-per-second and model time
            stats["detection_fps"] = round(0.8 * stats["detection_fps"] + 0.2 * (1 / loop_s), 1)
            if infer_ms is not None:
                prev = stats["inference_ms"]
                stats["inference_ms"] = round(infer_ms if prev is None else 0.8 * prev + 0.2 * infer_ms, 1)

        with detection_lock:
            # 🔥 If detected → update
            if label:
                latest_detection["label"] = label
                latest_detection["distance"] = distance
                latest_detection["last_seen"] = current_time

            # 🔥 If not detected → HOLD previous value
            elif current_time - latest_detection["last_seen"] > DETECTION_HOLD_TIME:
                latest_detection["label"] = "No traffic light detected"
                latest_detection["distance"] = None

            latest_detection["frame"] = frame

        time.sleep(0.05)


# =====================
# FETCH FUNCTION
# =====================
def detect_single_frame():
    with detection_lock:
        data = latest_detection.copy()

    return data["frame"], data["label"], data["distance"]