
import os
import sys
from google_pipeline2 import process_video

# Add ffmpeg to PATH (copied from google_pipeline2.py just in case, though it should be set by the module if imported)
os.environ["PATH"] += os.pathsep + r"C:\Users\anand\miniconda3\envs\tf_env\Library\bin"

def test_short_video():
    input_dir = "Videos/test"
    output_dir = "Video out"
    filename = "Copy of Chapter 2A - Green House and Poly House Types and Management.mp4"
    video_path = os.path.join(input_dir, filename)
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    print(f"Testing backend on {filename} with duration_limit=10s...")
    
    try:
        result = process_video(
            video_path=video_path, 
            output_dir=output_dir, 
            duration_limit=10, 
            input_lang="English", 
            output_lang="Hindi", 
            target_voice="Male (Chirp 3 - Alnilam)", 
            speed=1.0
        )
        print("\nTest Completed Successfully!")
        print(f"Output Video: {result['output_path']}")
        print(f"Source SRT:\n{result['source_srt'][:200]}...")
        print(f"Target SRT:\n{result['target_srt'][:200]}...")
        
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_short_video()
