
import requests
import os
import time

BASE_URL = "http://localhost:8002/api"
FILE_PATH = "Uploads/Copy of Chapter 2A - Green House and Poly House Types and Management_trimmed_10s.mp4"

def test_full_flow():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print(f"1. Uploading {FILE_PATH}...")
    try:
        with open(FILE_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            
        if response.status_code != 200:
            print(f"Upload failed: {response.text}")
            return
            
        data = response.json()
        filename = data["filename"]
        print(f"Upload success. Filename: {filename}")
        
        print("2. Starting Dubbing...")
        payload = {
            "filename": filename,
            "input_lang": "English",
            "output_lang": "Hindi",
            "target_voice": "Female",
            "speed": 1.0,
            "duration_limit": 10
        }
        
        dub_response = requests.post(f"{BASE_URL}/dub", json=payload)
        
        if dub_response.status_code != 200:
            print(f"Dub start failed: {dub_response.text}")
            return
            
        task_id = dub_response.json()["task_id"]
        print(f"Dubbing started. Task ID: {task_id}")
        
        print("3. Polling Status...")
        for _ in range(30): # Wait up to 30 seconds
            status_res = requests.get(f"{BASE_URL}/status/{task_id}")
            status_data = status_res.json()
            print(f"Status: {status_data['status']} | Step: {status_data['step']} | Progress: {status_data['progress']}%")
            
            if status_data['status'] in ['completed', 'failed']:
                break
            time.sleep(1)
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_full_flow()
