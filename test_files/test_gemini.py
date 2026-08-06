import sys
import os
import time

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("✅ Test started")

from gemini_assistant import get_gemini_response, speak

print("✅ gemini_assistant imported")

def main():
    print("🤖 Asking Gemini...")

    prompt = "What should a driver do when the traffic light is red?"

    # Gemini will speak internally (your function already calls speak)
    response = get_gemini_response(prompt)

    print("🗣️ Gemini said:", response)

    # Extra wait to ensure speech completes (IMPORTANT)
    time.sleep(6)

    print("✅ Test completed")

if __name__ == "__main__":
    main()
