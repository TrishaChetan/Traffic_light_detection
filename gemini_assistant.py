import os
import sys
import threading

import pyttsx3
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# sapi5 is Windows-only; let pyttsx3 pick nsss (macOS) / espeak (Linux) elsewhere
try:
    engine = pyttsx3.init(driverName="sapi5" if sys.platform == "win32" else None)
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)
except Exception as e:  # no TTS backend installed — keep running without voice
    print("⚠️ Text-to-speech unavailable:", e)
    engine = None

speech_lock = threading.Lock()
is_speaking = False
last_spoken_text = None


def _speak_blocking(text):
    global is_speaking
    with speech_lock:
        is_speaking = True
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            is_speaking = False


def speak(text):
    global last_spoken_text, is_speaking

    if engine is None:
        return

    # 🔥 prevent repeat
    if text == last_spoken_text:
        return

    # 🔥 do NOT interrupt current speech
    if is_speaking:
        return

    last_spoken_text = text

    threading.Thread(
        target=_speak_blocking,
        args=(text,),
        daemon=True
    ).start()


def get_gemini_response(prompt):
    try:
        prompt += " Please answer in 1-2 short sentences."

        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt
        )

        reply = response.text.strip()
        print("💬 Gemini:", reply)

        speak(reply)
        return reply

    except Exception as e:
        print("❌ Gemini error:", e)
        return "AI temporarily unavailable."