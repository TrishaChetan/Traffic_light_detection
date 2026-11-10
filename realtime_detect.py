import os
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
            time.sleep(0.01)

        print("⚠️ Camera stream lost — reconnecting")
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
            if latest_camera_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_camera_frame.copy()

        label = None
        distance = None

        try:
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            results = model(resized, conf=CONF_THRESHOLD, verbose=False)

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

        current_time = time.time()

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