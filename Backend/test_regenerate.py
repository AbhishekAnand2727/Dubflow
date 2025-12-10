
import requests
import json

BASE_URL = "http://localhost:8002/api"
TASK_ID = "0baca03a-adb5-4cbe-b512-0cf78c90329b" # The successful task from previous test

def test_regenerate():
    print(f"Testing regeneration for task: {TASK_ID}")
    
    # 1. Get current result to get segments
    res = requests.get(f"{BASE_URL}/result/{TASK_ID}")
    if res.status_code != 200:
        print(f"Failed to get result: {res.text}")
        return
        
    data = res.json()
    segments = data["source_segments"]
    
    # Modify a segment slightly to force regeneration
    if segments:
        segments[0]["text"] += " (Updated)"
        # Also add target_text to simulate frontend sending it
        segments[0]["target_text"] = "Updated Target Text"
        
    payload = {
        "task_id": TASK_ID,
        "segments": segments
    }
    
    print("Sending regenerate request...")
    res = requests.post(f"{BASE_URL}/regenerate", json=payload)
    
    if res.status_code != 200:
        print(f"Regeneration failed: {res.text}")
        return
        
    print("Regeneration started. Polling status...")
    
    import time
    for _ in range(30):
        status_res = requests.get(f"{BASE_URL}/status/{TASK_ID}")
        status_data = status_res.json()
        print(f"Status: {status_data['status']} | Step: {status_data['step']} | Progress: {status_data['progress']}%")
        
        if status_data['status'] in ['completed', 'failed']:
            if status_data['status'] == 'failed':
                print(f"Error: {status_data.get('error')}")
            break
        time.sleep(1)

if __name__ == "__main__":
    test_regenerate()
