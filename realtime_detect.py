import cv2
import threading
import time
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
MODEL_PATH = "best.pt"
CAMERA_URL = "http://192.168.31.37:8080/video"
IMG_SIZE = 320
K = 2000

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
def camera_loop():
    global latest_camera_frame

    cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Camera failed to open")
        return

    print("📷 Camera thread started")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.01)
            continue

        with camera_lock:
            latest_camera_frame = frame

        time.sleep(0.01)


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
            results = model(resized, conf=0.4, verbose=False)

            if results and len(results[0].boxes) > 0:
                box = results[0].boxes[0]

                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]

                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                box_width = xmax - xmin

                if box_width > 0:
                    distance = round(K / box_width, 1)

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