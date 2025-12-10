
import requests

try:
    response = requests.get("http://localhost:8002/api/languages")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error connecting to server: {e}")
