import requests
import time
import os

API_BASE = "http://localhost:8001/api"
FILE_PATH = "dubflow test.mp4"

def run_test():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print(f"Uploading {FILE_PATH}...")
    with open(FILE_PATH, 'rb') as f:
        files = {'file': f}
        try:
            res = requests.post(f"{API_BASE}/upload", files=files)
            res.raise_for_status()
            filename = res.json()['filename']
            print(f"Uploaded as: {filename}")
        except Exception as e:
            print(f"Upload failed: {e}")
            return

    print("Starting dubbing task (English -> Tamil)...")
    try:
        payload = {
            "filename": filename,
            "input_lang": "English",
            "output_lang": "Tamil",
            "target_voice": "Female",
            "speed": 1.0
        }
        res = requests.post(f"{API_BASE}/dub", json=payload)
        res.raise_for_status()
        task_id = res.json()['task_id']
        print(f"Task started with ID: {task_id}")
    except Exception as e:
        print(f"Dubbing request failed: {e}")
        return

    print("Polling for status...")
    while True:
        try:
            res = requests.get(f"{API_BASE}/status/{task_id}")
            data = res.json()
            status = data['status']
            progress = data.get('progress', 0)
            step = data.get('step', 'Unknown')
            
            print(f"Status: {status} | Step: {step} | Progress: {progress}%")
            
            if status == 'completed':
                print("Dubbing Completed Successfully!")
                print(f"Result: {data.get('result')}")
                break
            elif status == 'failed':
                print(f"Dubbing Failed: {data.get('error')}")
                break
            
            time.sleep(2)
        except Exception as e:
            print(f"Polling failed: {e}")
            break

if __name__ == "__main__":
    run_test()
