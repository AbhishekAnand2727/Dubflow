
import sys
import os

# Ensure we can import google_pipeline2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import google_pipeline2

def test_voice_manager():
    vm = google_pipeline2.VoiceManager()
    
    # Test Case: Hindi + Fenrir
    target_voice = "Male (Chirp 3 - Fenrir)"
    output_lang = "Hindi"
    
    voice_id = vm.get_voice(speaker_label="Speaker 1", voice_preference=target_voice, output_lang=output_lang)
    print(f"Input: Voice='{target_voice}', Lang='{output_lang}'")
    print(f"Output Voice ID: {voice_id}")
    
    # Check if Fenrir is in personalities
    print(f"Is Fenrir in personalities? {'Fenrir' in vm.chirp_3_personalities}")

    # Try generating audio
    print("Attempting to generate audio with Fenrir...")
    try:
        api_key = google_pipeline2.get_api_key()
        audio = google_pipeline2.generate_google_tts("नमस्ते, यह एक परीक्षण है।", api_key, voice_name=voice_id, speaking_rate=1.0)
        if audio:
            print("Success! Audio generated.")
        else:
            print("Failed to generate audio (returned None).")
    except Exception as e:
        print(f"Error generating audio: {e}")

if __name__ == "__main__":
    test_voice_manager()
