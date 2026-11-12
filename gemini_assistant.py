import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import pyttsx3
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "5"))  # seconds — advice is useless if it's late
_client = None
_pool = ThreadPoolExecutor(max_workers=2)


def _get_client():
    """Create the Gemini client lazily, so a missing key doesn't stop the app starting."""
    global _client
    if _client is None and os.getenv("GEMINI_API_KEY"):
        try:
            _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        except Exception as e:
            print("⚠️ Gemini unavailable:", e)
    return _client

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


def rule_based_advice(label, distance=None):
    """Offline driving advice — used when Gemini has no key, errors, or is too slow."""
    colour = next((c for c in ("red", "yellow", "green") if c in (label or "").lower()), None)
    where = f"about {distance:g} metres ahead" if isinstance(distance, (int, float)) else "ahead"
    near = isinstance(distance, (int, float)) and distance < 15
    if colour == "red":
        return (f"Red light {where}. Brake now and stop before the line." if near
                else f"Red light {where}. Ease off and prepare to stop.")
    if colour == "yellow":
        return (f"Yellow light {where}. Stop if you safely can; don't speed up." if not near
                else f"Yellow light {where}. If you can't stop safely, proceed with care.")
    if colour == "green":
        return f"Green light {where}. Proceed carefully and watch for pedestrians."
    return "No traffic light detected. Drive carefully."


def _ask_gemini(prompt):
    client = _get_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY not set")
    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt + " Please answer in 1-2 short sentences."
    )
    reply = (response.text or "").strip()
    if not reply:
        raise RuntimeError("empty response")
    return reply


def get_driving_advice(label, distance=None):
    """Gemini advice with a time limit, falling back to built-in rules. Spoken aloud."""
    where = f"about {distance:g} meters" if isinstance(distance, (int, float)) else "nearby"
    prompt = f"There is a {label} signal {where} ahead. What should I do?"
    try:
        reply = _pool.submit(_ask_gemini, prompt).result(timeout=GEMINI_TIMEOUT)
        print("💬 Gemini:", reply)
    except FutureTimeout:
        reply = rule_based_advice(label, distance)
        print(f"⏱️ Gemini took over {GEMINI_TIMEOUT:g}s — using built-in advice:", reply)
    except Exception as e:
        reply = rule_based_advice(label, distance)
        print(f"ℹ️ Gemini unavailable ({e}) — using built-in advice:", reply)
    speak(reply)
    return reply


def get_gemini_response(prompt):
    """Kept for the scripts in test_files/: free-form prompt, no fallback rules."""
    try:
        reply = _ask_gemini(prompt)
        speak(reply)
        return reply
    except Exception as e:
        print("❌ Gemini error:", e)
        return "AI temporarily unavailable."