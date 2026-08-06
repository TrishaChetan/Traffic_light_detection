import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("✅ Test file started")

from realtime_detect import detect_traffic_light_live

print("✅ detect_traffic_light_live imported")

def main():
    detect_traffic_light_live()

if __name__ == "__main__":
    main()
