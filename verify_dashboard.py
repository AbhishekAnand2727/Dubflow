import requests
import json

API_BASE = "http://localhost:8001/api"

def test_dashboard_api():
    print(f"Testing API at {API_BASE}...")

    # 1. Get History
    try:
        res = requests.get(f"{API_BASE}/history")
        res.raise_for_status()
        history = res.json()
        print(f"[OK] /api/history returned {len(history)} tasks.")
        
        if not history:
            print("[FAIL] No tasks found in history! Dashboard will be empty.")
            return

        # 2. Pick a restored task
        task = history[0]
        task_id = task['id']
        print(f"Checking task: {task.get('filename')} (ID: {task_id})")
        print(f"  - Status: {task.get('status')}")
        print(f"  - Lang: {task.get('input_lang')} -> {task.get('output_lang')}")

        # 3. Get Result Details (Transcripts)
        res_details = requests.get(f"{API_BASE}/result/{task_id}")
        res_details.raise_for_status()
        details = res_details.json()
        
        source_segs = details.get('source_segments', [])
        target_segs = details.get('target_segments', [])
        
        print(f"[OK] /api/result/{task_id} returned details.")
        print(f"  - Source Segments: {len(source_segs)}")
        print(f"  - Target Segments: {len(target_segs)}")
        
        if len(source_segs) > 0:
            print(f"  - Sample Source: {source_segs[0]}")
        if len(target_segs) > 0:
            print(f"  - Sample Target: {target_segs[0]}")

        if len(source_segs) > 0 and len(target_segs) > 0:
            print("[OK] Transcripts are present and ready for the interactive player.")
        else:
            print("[WARN] Transcripts are missing or empty.")

    except Exception as e:
        print(f"[FAIL] API Test Failed: {e}")

if __name__ == "__main__":
    test_dashboard_api()
