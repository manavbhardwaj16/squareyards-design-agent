import urllib.request
import json
import time

payload = json.dumps({"brief_id": "BR-01"}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/run-agent",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print("Sending POST /run-agent with {'brief_id': 'BR-01'}...")
start = time.time()
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        duration = round(time.time() - start, 2)
        print(f"HTTP {resp.status} received in {duration}s")
        data = json.loads(resp.read().decode())
        print("=== RESPONSE RECEIVED ===")
        print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)
