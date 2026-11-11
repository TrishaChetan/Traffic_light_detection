from flask import Flask, render_template, Response, jsonify
import threading
import time
from collections import deque

import cv2

from realtime_detect import camera_loop, detection_loop, detect_single_frame
from main import analyze_auto

app = Flask(__name__)

# =====================
# GLOBAL STATE
# =====================
NO_LIGHT = "No traffic light detected"

latest_data = {
    "label": NO_LIGHT,
    "distance": None,
    "response": "Starting system...",
    "updated_at": None,
    "history": [],
}

# last 10 light changes, newest first, for the dashboard timeline
history = deque(maxlen=10)

data_lock = threading.Lock()


# =====================
# START THREADS
# =====================
def start_threads():
    print("📷 Starting camera thread")
    threading.Thread(target=camera_loop, daemon=True).start()

    print("🚀 Starting detection thread")
    threading.Thread(target=detection_loop, daemon=True).start()

    print("🧠 Starting AI update thread")
    threading.Thread(target=update_data_loop, daemon=True).start()


# =====================
# UPDATE LOOP (FIXED)
# =====================
def update_data_loop():
    global latest_data

    while True:
        try:
            # analyze_auto() applies the 3 s hold, so the label clears when the
            # light really disappears instead of sticking on the last one forever
            frame, label, distance, response = analyze_auto()

            if frame is None:
                time.sleep(0.1)
                continue

            with data_lock:
                if label != latest_data["label"]:
                    history.appendleft({
                        "time": time.strftime("%H:%M:%S"),
                        "label": label,
                        "distance": distance,
                    })
                    print("🚦 Light changed →", label, distance)

                latest_data["label"] = label
                latest_data["distance"] = distance
                latest_data["response"] = response
                latest_data["updated_at"] = time.time()
                latest_data["history"] = list(history)

        except Exception as e:
            print("⚠️ Update loop error:", e)

        time.sleep(0.1)


# =====================
# VIDEO STREAM
# =====================
def generate():
    print("📡 Video stream started")

    while True:
        try:
            frame, _, _ = detect_single_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)

        except Exception as e:
            print("⚠️ Stream error:", e)
            time.sleep(0.1)


# =====================
# ROUTES
# =====================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/latest_detection')
def latest_detection():
    with data_lock:
        return jsonify(latest_data)


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    start_threads()
    app.run(debug=True, use_reloader=False)