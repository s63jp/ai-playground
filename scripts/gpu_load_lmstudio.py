"""Keep LM Studio generating on the GPU for N seconds (to test XPU training under GPU contention)."""
import sys
import time

import httpx

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 180
model = sys.argv[2] if len(sys.argv) > 2 else "unslothqen 4b"
end = time.time() + secs
n = toks = 0
while time.time() < end:
    try:
        r = httpx.post("http://127.0.0.1:1234/v1/chat/completions", timeout=120, json={
            "model": model, "max_tokens": 400, "temperature": 0.8,
            "messages": [{"role": "user", "content": "Write a long, detailed story about an island. Keep going."}]})
        toks += r.json().get("usage", {}).get("completion_tokens", 0)
        n += 1
    except Exception as e:  # noqa: BLE001
        print("request error:", type(e).__name__, e, flush=True)
        time.sleep(1)
print(f"LOAD DONE: {n} completions, {toks} tokens in {secs:.0f}s", flush=True)
