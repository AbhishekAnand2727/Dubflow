
import sys
import os
import time

# Ensure we can import google_pipeline2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import google_pipeline2

def test_speed():
    api_key = google_pipeline2.get_api_key()
    voice_name = "hi-IN-Chirp3-HD-Fenrir"
    text = "नमस्ते, यह एक गति परीक्षण है। मैं देख रहा हूँ कि क्या मैं धीरे बोल सकता हूँ।"
    
    print(f"Testing Voice: {voice_name}")
    
    # Speed 1.0
    print("Generating at Speed 1.0...")
    audio_1 = google_pipeline2.generate_google_tts(text, api_key, voice_name=voice_name, speaking_rate=1.0)
    len_1 = len(audio_1) if audio_1 else 0
    print(f"Length at 1.0: {len_1} bytes")
    
    # Speed 0.5
    print("Generating at Speed 0.5...")
    audio_2 = google_pipeline2.generate_google_tts(text, api_key, voice_name=voice_name, speaking_rate=0.5)
    len_2 = len(audio_2) if audio_2 else 0
    print(f"Length at 0.5: {len_2} bytes")
    
    if len_2 > len_1:
        print("SUCCESS: Audio at 0.5 is longer.")
        print(f"Ratio: {len_2/len_1:.2f}")
    else:
        print("FAILURE: Audio at 0.5 is NOT longer.")

if __name__ == "__main__":
    test_speed()
