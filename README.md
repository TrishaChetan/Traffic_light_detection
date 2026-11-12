# 🚦 Traffic Light Detection + AI Voice Assistant

Real-time traffic light detection with a custom-trained **YOLO** model, a live web dashboard, and a **Gemini**-powered voice assistant that tells the driver what to do ("Red light about 12 m ahead — slow down and stop").

## How it works

```
Phone camera / webcam ──► camera thread ──► YOLO (best.pt) ──► label + distance ──► Flask dashboard
                          (auto-reconnect)   red / yellow / green      │
                                                                       └──► Gemini ──► spoken advice (TTS)
```

- **Detection** — `best.pt` is a YOLO model trained on three classes: `red_light`, `yellow_light`, `green_light`. The most confident box in each frame is used and drawn on the live video.
- **Distance estimate** — approximated from bounding-box width (`distance ≈ K / box_width`, `K = 2000` in `realtime_detect.py`; calibrate for your camera).
- **Stability** — a detection is held for 3 s so the label doesn't flicker when a frame is missed.
- **AI assistant** — when the light changes, Gemini is asked for a one-sentence driving instruction, which is spoken aloud (with a cooldown so it doesn't repeat itself).
- **Always answers** — if there's no `GEMINI_API_KEY`, Gemini errors (e.g. quota) or takes longer than `GEMINI_TIMEOUT` seconds (default 5), a built-in rule gives the instruction instead, based on the light colour and distance. The app also starts and works fully offline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your-key-here
# A phone running the "IP Webcam" app, or 0 for your laptop webcam
CAMERA_SOURCE=http://192.168.1.20:8080/video
# Optional
GEMINI_TIMEOUT=5
CONF_THRESHOLD=0.4
MODEL_PATH=best.pt
```

Run it:

```bash
python app.py
```

Open http://127.0.0.1:5000 to see the live feed with detections and the assistant's latest advice.

> **Voice on Linux:** text-to-speech uses `espeak` — install it with `sudo apt install espeak`. On Windows (SAPI5) and macOS it works out of the box. If no voice engine is available the app keeps running silently.

## Training your own model

1. Put images in `dataset/train_dataset/train_images` and annotations in `train.json`.
2. Convert annotations to YOLO format: `python convert_json_to_yolo.py`
3. Train: `yolo detect train data=data.yaml model=yolov8n.pt imgsz=320 epochs=50`
4. Copy `runs/detect/train/weights/best.pt` to the project root.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Flask server: video stream, `/latest_detection` JSON endpoint, background threads |
| `realtime_detect.py` | Camera thread (with reconnect) and YOLO detection thread |
| `main.py` | Detection stabilisation and when to ask Gemini |
| `gemini_assistant.py` | Gemini API call + cross-platform text-to-speech |
| `convert_json_to_yolo.py` | Converts JSON annotations to YOLO label files |
| `templates/index.html` | Dashboard UI |
| `test_files/` | Standalone tests for the model and the Gemini call |

## Safety note

This is a student/research prototype. It is not a certified driver-assistance system — never rely on it to make driving decisions.
