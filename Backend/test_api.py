
import requests
import json

base_url = "http://localhost:8002/api"

tasks_to_check = [
    "7fbe453c-42d7-4a3c-b131-8d0e7a4911a2", # Completed
    "Copy of Chapter 2A - Green House and Poly House Types and Management_trimmed_10s" # Failed
]

for task_id in tasks_to_check:
    print(f"Checking task: {task_id}")
    try:
        response = requests.get(f"{base_url}/result/{task_id}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Success")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    print("-" * 20)
