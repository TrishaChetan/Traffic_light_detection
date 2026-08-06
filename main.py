from realtime_detect import detect_single_frame
from gemini_assistant import get_gemini_response
import time
import threading

# =====================
# STATE CONTROL
# =====================
last_spoken_label = None
last_spoken_time = 0
COOLDOWN = 10

# =====================
# DETECTION PERSISTENCE
# =====================
last_detected_label = "No traffic light detected"
last_detected_distance = None
last_detection_time = 0
DETECTION_HOLD_TIME = 3  # seconds

latest_response = "System ready."


# =====================
# ASYNC GEMINI CALL
# =====================
def async_gemini_call(prompt):
    global latest_response
    try:
        response = get_gemini_response(prompt)
        latest_response = response
    except Exception as e:
        print("⚠️ Gemini async error:", e)


# =====================
# MAIN ANALYSIS
# =====================
def analyze_auto():
    global last_spoken_label, last_spoken_time
    global last_detected_label, last_detected_distance, last_detection_time
    global latest_response

    frame, label, distance = detect_single_frame()
    current_time = time.time()

    if frame is None:
        return None, last_detected_label, last_detected_distance, latest_response

    # =====================
    # 🔥 DETECTION STABILITY FIX
    # =====================
    if label:
        last_detected_label = label
        last_detected_distance = distance
        last_detection_time = current_time

    elif current_time - last_detection_time > DETECTION_HOLD_TIME:
        # only reset after hold time
        last_detected_label = "No traffic light detected"
        last_detected_distance = None

    # Always use stable values
    stable_label = last_detected_label
    stable_distance = last_detected_distance

    response = latest_response

    # =====================
    # 🔥 GEMINI TRIGGER
    # =====================
    if label != "No traffic light detected":
        distance_text = f"{stable_distance} meters" if stable_distance else "nearby"

        if (
            stable_label != last_spoken_label and
            (current_time - last_spoken_time > COOLDOWN)
        ):
            prompt = f"There is a {stable_label} signal {distance_text} ahead. What should I do?"

            threading.Thread(
                target=async_gemini_call,
                args=(prompt,),
                daemon=True
            ).start()

            last_spoken_label = stable_label
            last_spoken_time = current_time

    return frame, stable_label, stable_distance, response