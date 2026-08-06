import os
import pyttsx3
import threading
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

engine = pyttsx3.init(driverName="sapi5")
engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)

speech_lock = threading.Lock()
is_speaking = False
last_spoken_text = None


def _speak_blocking(text):
    global is_speaking
    with speech_lock:
        is_speaking = True

        engine.say(text)
        engine.runAndWait()

        is_speaking = False


def speak(text):
    global last_spoken_text, is_speaking

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